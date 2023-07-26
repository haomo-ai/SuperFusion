import argparse
import tqdm

import torch

from data.dataset_front import semantic_dataset
from data.const import NUM_CLASSES
from model_front import get_model


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def get_batch_iou(pred_map, gt_map):
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[1]):
            pred = pred_map[:, i]
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float()
            union = (pred | tgt).sum().float()
            intersects.append(intersect)
            unions.append(union)
    return torch.tensor(intersects), torch.tensor(unions)

def eval_iou(model, val_loader):
    model.eval()
    total_intersects = 0
    total_union = 0
    total_intersects_split_30_60 = 0
    total_union_split_30_60 = 0
    total_intersects_split_60_90 = 0
    total_union_split_60_90 = 0
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, final_depth_map, final_depth_map_bin_enc, projected_depth in tqdm.tqdm(val_loader):


            semantic, embedding, direction, _ = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), final_depth_map_bin_enc.cuda(), projected_depth.cuda())
            
            semantic_gt = semantic_gt.cuda().float()
            split = int(semantic_gt.shape[3]/3)

            intersects, union = get_batch_iou(
                onehot_encoding(semantic[:,:,:,:split]), semantic_gt[:,:,:,:split])
            total_intersects += intersects
            total_union += union


            intersects, union = get_batch_iou(
                onehot_encoding(semantic[:,:,:,split:2*split]), semantic_gt[:,:,:,split:2*split])
            total_intersects_split_30_60 += intersects
            total_union_split_30_60 += union

            intersects, union = get_batch_iou(
                onehot_encoding(semantic[:,:,:,2*split:]), semantic_gt[:,:,:,2*split:])
            total_intersects_split_60_90 += intersects
            total_union_split_60_90 += union
    return total_intersects / (total_union + 1e-7), total_intersects_split_30_60 / (total_union_split_30_60 + 1e-7), total_intersects_split_60_90 / (total_union_split_60_90 + 1e-7)


def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'depth_image_size': args.depth_image_size,
    }
    train_loader, val_loader = semantic_dataset(
        args.version, args.dataroot, data_conf, args.bsz, args.nworkers, depth_downsample_factor=args.depth_downsample_factor, depth_sup=args.depth_sup, use_depth_enc=args.use_depth_enc, use_depth_enc_bin=args.use_depth_enc_bin, add_depth_channel=args.add_depth_channel,use_lidar_10=args.use_lidar_10)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim,
                      args.direction_pred, args.angle_class, downsample=args.depth_downsample_factor, use_depth_enc=args.use_depth_enc, pretrained=args.pretrained, add_depth_channel=args.add_depth_channel,add_fuser=args.add_fuser)
    
    if args.model == 'HDMapNet_fusion' or args.model == 'HDMapNet_cam':
        model.load_state_dict(torch.load(args.modelf), strict=False)
    else:
        checkpoint = torch.load(args.modelf)
        model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    iou_front, iou_back, iou_60_90 = eval_iou(model, val_loader)
    print("iou_0_30: ", iou_front)
    print("iou_30_60: ", iou_back)
    print("iou_60_90: ", iou_60_90)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # nuScenes config
    parser.add_argument('--dataroot', type=str,
                        default='/path/to/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='SuperFusion')

    # training config
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)

    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[256, 704])
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
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    parser.add_argument('--lidar_cut_x', action='store_true')
    parser.add_argument("--TOP_X_MIN", type=int, default=-20)
    parser.add_argument("--TOP_X_MAX", type=int, default=20)
    parser.add_argument("--camC", type=int, default=64)
    parser.add_argument("--lidarC", type=int, default=128)
    parser.add_argument("--crossC", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument('--cross_atten', action='store_true')
    parser.add_argument('--cross_conv', action='store_true')
    parser.add_argument('--add_bn', action='store_true')
    parser.add_argument('--pos_emd', action='store_true')
    parser.add_argument('--pos_emd_img', action='store_true')
    parser.add_argument('--lidar_feature_trans', action='store_true')

    parser.add_argument("--depth_downsample_factor", type=int, default=4)
    parser.add_argument('--depth_sup', action='store_true')
    parser.add_argument("--depth_image_size", nargs=2, type=int, default=[256, 704])


    parser.add_argument('--lidar_pred', action='store_true')
    parser.add_argument('--use_cross', action='store_true')
    parser.add_argument('--add_fuser', action='store_true')
    parser.add_argument('--add_fuser2', action='store_true')
    parser.add_argument('--use_depth_enc', action='store_true')
    parser.add_argument('--add_depth_channel', action='store_true')
    parser.add_argument('--use_depth_enc_bin', action='store_true')
    parser.add_argument('--add_fuser_AlignFA', action='store_true')
    parser.add_argument('--add_fuser_AlignFAnew', action='store_true')
    parser.add_argument('--use_lidar_10', action='store_true')
    parser.add_argument('--pretrained', action='store_true')

    args = parser.parse_args()
    main(args)
