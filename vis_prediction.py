import argparse
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import tqdm
import torch
import math
from data.dataset_front import semantic_dataset
from data.const import NUM_CLASSES
from model_front import get_model
from postprocess.vectorize import vectorize


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def vis_segmentation(model, val_loader):
    model.eval()
    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(val_loader):

            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                   post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                   lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            semantic = semantic.softmax(1).cpu().numpy()
            semantic[semantic_gt < 0.1] = np.nan

            for si in range(semantic.shape[0]):
                plt.figure(figsize=(4, 2))
                plt.imshow(semantic[si][1], vmin=0,
                           cmap='Blues', vmax=1, alpha=0.6)
                plt.imshow(semantic[si][2], vmin=0,
                           cmap='Reds', vmax=1, alpha=0.6)
                plt.imshow(semantic[si][3], vmin=0,
                           cmap='Greens', vmax=1, alpha=0.6)

                # fig.axes.get_xaxis().set_visible(False)
                # fig.axes.get_yaxis().set_visible(False)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                plt.axis('off')

                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(imname)
                plt.close()


def vis_vector(model, val_loader, angle_class):
    # model.eval()
    car_img = Image.open('pics/car.png')

    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt, final_depth_map_bin, final_depth_map_bin_enc, projected_depth, vectors, rec) in enumerate(val_loader):

            segmentation, embedding, direction, depth = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                       post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                       lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), final_depth_map_bin_enc.cuda(), projected_depth.cuda())

            # for si in range(segmentation.shape[0]):
            for si in range(1):
                coords, _, _ = vectorize(
                    segmentation[si], embedding[si], direction[si], angle_class)

                for coord in coords:
                    plt.plot(coord[:, 0], coord[:, 1])

                plt.xlim((0, segmentation.shape[3]))
                plt.ylim((0, segmentation.shape[2]))

                plt.imshow(car_img, extent=[
                           -15, 15, segmentation.shape[2]//2-12, segmentation.shape[2]//2+12])

                img_name = 'results/'+args.saveroot + \
                    f'/eval_{batchi:04}_'+str(rec['data']['CAM_FRONT'])+'_pred.jpg'
                print('saving', img_name)
                # plt.savefig(img_name)
                plt.axis('off')
                plt.savefig(img_name, bbox_inches='tight', dpi=400)
                plt.close()


def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'depth_image_size': args.depth_image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }

    train_loader, val_loader = semantic_dataset(
        args.version, args.dataroot, data_conf, args.bsz, args.nworkers, depth_downsample_factor=args.depth_downsample_factor, depth_sup=args.depth_sup, use_depth_enc=args.use_depth_enc, use_depth_enc_bin=args.use_depth_enc_bin, add_depth_channel=args.add_depth_channel,use_lidar_10=True, visual=True)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim,
                      args.direction_pred, args.angle_class, downsample=args.depth_downsample_factor, use_depth_enc=args.use_depth_enc, pretrained=args.pretrained, add_depth_channel=args.add_depth_channel,add_fuser=args.add_fuser)
    # model.load_state_dict(torch.load(args.modelf), strict=False)

    checkpoint = torch.load(args.modelf)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    vis_vector(model, val_loader, args.angle_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # nuScenes config
    parser.add_argument('--dataroot', type=str,
                        default='/media/hao/HaoData/dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='SuperFusion')

    # training config
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=10)

    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--depth_downsample_factor", type=int, default=4)
    parser.add_argument("--image_size", nargs=2, type=int, default=[256, 704])
    parser.add_argument("--depth_image_size", nargs=2, type=int, default=[256, 704])
    parser.add_argument("--xbound", nargs=3, type=float,
                        default=[-90.0, 90.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float,
                        default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float,
                        default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float,
                        default=[2.0, 90.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    parser.add_argument("--camC", type=int, default=64)
    parser.add_argument("--lidarC", type=int, default=128)
    parser.add_argument("--crossC", type=int, default=128)

    parser.add_argument('--saveroot', type=str,
                        default='SuperFusion')

    parser.add_argument('--depth_sup', action='store_true')
    parser.add_argument('--use_depth_enc', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--use_depth_enc_bin', action='store_true')
    parser.add_argument('--add_depth_channel', action='store_true')
    parser.add_argument('--add_fuser', action='store_true')


    args = parser.parse_args()
    main(args)
