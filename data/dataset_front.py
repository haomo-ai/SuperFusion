import os
import numpy as np

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from torch.utils.data import Dataset
from data.rasterize import preprocess_map
from .const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W
from .vector_map import VectorizedLocalMap
from .lidar import get_lidar_data
from .image import normalize_img, img_transform, normalize_img_aug, normalize_img_aug_flip
from .utils import label_onehot_encoding, ego_to_cam_points, project_depths, bin_depths
from .ip_basic import fill_in_multiscale
from model_front.voxel import pad_or_trim_to_np
import skimage.transform
import random


class HDMapNetDataset(Dataset):
    def __init__(self, version, dataroot, data_conf, is_train, lidar_cut_x=False, TOP_X_MIN=-20, TOP_X_MAX=20, mask_lidar=False):
        super(HDMapNetDataset, self).__init__()
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]
        canvas_h = int(patch_h / data_conf['ybound'][2])
        canvas_w = int(patch_w / data_conf['xbound'][2])
        self.is_train = is_train
        self.data_conf = data_conf
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.vector_map = VectorizedLocalMap(
            dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)
        self.scenes = self.get_scenes(version, is_train)
        self.samples = self.get_samples()
        self.lidar_cut_x = lidar_cut_x
        self.TOP_X_MIN = TOP_X_MIN
        self.TOP_X_MAX = TOP_X_MAX
        self.mask_lidar = mask_lidar

    def __len__(self):
        return len(self.samples)

    def get_scenes(self, version, is_train):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[version][is_train]

        return create_splits_scenes()[split]

    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_lidar(self, rec, data_aug, flip_sign):
        lidar_data = get_lidar_data(
            self.nusc, rec, nsweeps=3, min_distance=2.2)
        lidar_data_original = torch.Tensor(lidar_data)[:3]
        lidar_data = lidar_data.transpose(1, 0)

        DA = False
        drop_points = False
        if data_aug:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                drop_points = random.uniform(0, 0.2)

        num_points = lidar_data.shape[0]

        if drop_points is not False:
            points_to_drop = np.random.randint(
                0, num_points-1, int(num_points*drop_points))
            lidar_data = np.delete(
                lidar_data, points_to_drop, axis=0)

        if flip_sign:
            lidar_data[:, 1] = -lidar_data[:, 1]

        if DA:
            jitter_x = random.uniform(-0.05, 0.05)
            jitter_y = random.uniform(-0.05, 0.05)
            jitter_z = random.uniform(-1.0, 1.0)
            lidar_data[:, 0] += jitter_x
            lidar_data[:, 1] += jitter_y
            lidar_data[:, 2] += jitter_z

        lidar_data = pad_or_trim_to_np(
            lidar_data, [81920, 5]).astype('float32')
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0
        return lidar_data, lidar_mask, lidar_data_original

    def get_lidar_10(self, rec, nsweeps=10):
        lidar_data = get_lidar_data(
            self.nusc, rec, nsweeps=nsweeps, min_distance=2.2)
        lidar_data_original = torch.Tensor(lidar_data)[:3]
        return lidar_data_original

    def get_ego_pose(self, rec):
        sample_data_record = self.nusc.get(
            'sample_data', rec['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get(
            'ego_pose', sample_data_record['ego_pose_token'])
        car_trans = ego_pose['translation']
        pos_rotation = Quaternion(ego_pose['rotation'])
        yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)

    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size']
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)
        resize_dims = (fW, fH)
        return resize, resize_dims

    def get_imgs(self, rec, data_aug, flip_sign):
        imgs = []
        trans = []
        rots = []
        intrins = []
        post_trans = []
        post_rots = []
        cam = 'CAM_FRONT'

        samp = self.nusc.get('sample_data', rec['data'][cam])
        imgname = os.path.join(self.nusc.dataroot, samp['filename'])
        img = Image.open(imgname)

        resize, resize_dims = self.sample_augmentation()
        img, post_rot, post_tran = img_transform(img, resize, resize_dims)

        if data_aug and flip_sign:
            img = normalize_img_aug_flip(img)
        elif data_aug:
            img = normalize_img_aug(img)
        else:
            img = normalize_img(img)
        post_trans.append(post_tran)
        post_rots.append(post_rot)
        imgs.append(img)

        sens = self.nusc.get('calibrated_sensor',
                             samp['calibrated_sensor_token'])
        trans.append(torch.Tensor(sens['translation']))
        rots.append(torch.Tensor(Quaternion(
            sens['rotation']).rotation_matrix))
        intrins.append(torch.Tensor(sens['camera_intrinsic']))
        return torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(post_trans), torch.stack(post_rots)

    def get_vectors(self, rec):
        location = self.nusc.get('log', self.nusc.get(
            'scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get(
            'sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = self.vector_map.gen_vectorized_samples(
            location, ego_pose['translation'], ego_pose['rotation'])
        return vectors

    def __getitem__(self, idx):
        rec = self.samples[idx]
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask, lidar_data_original = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        vectors = self.get_vectors(rec)

        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, vectors


class HDMapNetSemanticDataset(HDMapNetDataset):
    def __init__(self, version, dataroot, data_conf, is_train, lidar_cut_x=False, TOP_X_MIN=-20, TOP_X_MAX=20, visual=False, depth_downsample_factor = 16, depth_sup=True, use_depth_enc=False, use_depth_enc_bin=False, add_depth_channel=False, use_lidar_10=False, data_aug=False):
        super(HDMapNetSemanticDataset, self).__init__(
            version, dataroot, data_conf, is_train, lidar_cut_x, TOP_X_MIN, TOP_X_MAX)
        self.thickness = data_conf['thickness']
        self.angle_class = data_conf['angle_class']
        self.visual = visual

        fH, fW = self.data_conf['depth_image_size']
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)
        post_rot2 = torch.eye(2)

        rot_resize = torch.Tensor([[resize[0], 0],
                                [0, resize[1]]])
        post_rot2 = rot_resize @ post_rot2

        self.post_rot = torch.eye(3)
        self.post_rot[:2, :2] = post_rot2
        self.depth_final_dim=(fH, fW)
        self.depth_downsample_factor = depth_downsample_factor
        self.depth_sup = depth_sup
        self.use_depth_enc = use_depth_enc
        self.use_depth_enc_bin = use_depth_enc_bin
        self.add_depth_channel = add_depth_channel
        self.use_lidar_10 = use_lidar_10
        self.data_aug = data_aug

    def get_semantic_map(self, rec):
        vectors = self.get_vectors(rec)
        instance_masks, forward_masks, backward_masks = preprocess_map(
            vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness, self.angle_class)
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat(
            [(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
        instance_masks = instance_masks.sum(0)
        forward_oh_masks = label_onehot_encoding(
            forward_masks, self.angle_class+1)
        backward_oh_masks = label_onehot_encoding(
            backward_masks, self.angle_class+1)
        direction_masks = forward_oh_masks + backward_oh_masks
        direction_masks = direction_masks / direction_masks.sum(0)
        return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks, vectors

    def __getitem__(self, idx):
        flip_sign = False
        if self.data_aug:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    flip_sign = True

        rec = self.samples[idx]
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec, self.data_aug, flip_sign)
        lidar_data, lidar_mask, lidar_data_original = self.get_lidar(rec, self.data_aug, flip_sign)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        semantic_masks, instance_masks, _, _, direction_masks, vectors = self.get_semantic_map(
            rec)
        if self.depth_sup:
            cam_p = self.post_rot @ intrins[0]
            cam_pts = ego_to_cam_points(self.get_lidar_10(rec), rots[0], trans[0])
            final_depth_map, _ = fill_in_multiscale(project_depths(cam_pts, cam_p, self.depth_final_dim))
            final_depth_map = skimage.transform.downscale_local_mean(image=final_depth_map,
                                                        factors=(self.depth_downsample_factor, self.depth_downsample_factor))
            final_depth_map_bin = bin_depths(final_depth_map, self.data_conf['dbound'][0], self.data_conf['dbound'][1], self.data_conf['dbound'][2], target=True)

            final_depth_map_bin_enc = 0

            if self.add_depth_channel:
                cam_pts = ego_to_cam_points(lidar_data_original, rots[0], trans[0])
                projected_depths = project_depths(cam_pts, cam_p, self.depth_final_dim)
                projected_depth = np.expand_dims(projected_depths, axis=0).astype('float32')
            else:
                projected_depth = 0
        else:
            final_depth_map_bin = 0
            final_depth_map_bin_enc = 0
            projected_depth = 0


        if flip_sign:
            if self.depth_sup:
                final_depth_map_bin = torch.flip(final_depth_map_bin, dims=[0])
                if self.add_depth_channel:
                    projected_depth = np.flip(projected_depth, 1).copy()
            semantic_masks = torch.flip(semantic_masks, dims=[1])
            direction_masks = torch.flip(direction_masks, dims=[1])
            instance_masks = torch.flip(instance_masks, dims=[0])

        semantic_masks = semantic_masks[:, :,
                                        int(semantic_masks.shape[2]/2):]
        instance_masks = instance_masks[:, int(
            instance_masks.shape[1]/2):]
        direction_masks = direction_masks[:,
                                          :, int(direction_masks.shape[2]/2):]

        if self.visual:
            return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_masks, final_depth_map_bin, final_depth_map_bin_enc, projected_depth, vectors, rec
        else:
            return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_masks, final_depth_map_bin, final_depth_map_bin_enc, projected_depth


def semantic_dataset(version, dataroot, data_conf, bsz, nworkers, lidar_cut_x=False, TOP_X_MIN=-20, TOP_X_MAX=20, visual=False, depth_downsample_factor=16, depth_sup=True, use_depth_enc=False, use_depth_enc_bin=False, add_depth_channel=False, use_lidar_10=False, data_aug=False,data_seed=False):
    train_dataset = HDMapNetSemanticDataset(
        version, dataroot, data_conf, is_train=True, lidar_cut_x=lidar_cut_x, TOP_X_MIN=TOP_X_MIN, TOP_X_MAX=TOP_X_MAX, visual=visual, depth_downsample_factor=depth_downsample_factor, depth_sup=depth_sup, use_depth_enc=use_depth_enc, use_depth_enc_bin=use_depth_enc_bin, add_depth_channel=add_depth_channel, use_lidar_10=use_lidar_10, data_aug=data_aug)
    val_dataset = HDMapNetSemanticDataset(
        version, dataroot, data_conf, is_train=False, lidar_cut_x=lidar_cut_x, TOP_X_MIN=TOP_X_MIN, TOP_X_MAX=TOP_X_MAX, visual=visual, depth_downsample_factor=depth_downsample_factor, depth_sup=depth_sup, use_depth_enc=use_depth_enc, use_depth_enc_bin=use_depth_enc_bin, add_depth_channel=add_depth_channel, use_lidar_10=use_lidar_10, data_aug=False)

    if data_seed:
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True,
            worker_init_fn=seed_worker,
            generator=g)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_loader, val_loader


if __name__ == '__main__':
    data_conf = {
        'image_size': (900, 1600),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }

    dataset = HDMapNetSemanticDataset(
        version='v1.0-mini', dataroot='dataset/nuScenes', data_conf=data_conf, is_train=False)
    for idx in range(dataset.__len__()):
        imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_mask = dataset.__getitem__(
            idx)
