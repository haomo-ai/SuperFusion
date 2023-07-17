import os
import numpy as np
import sys
import logging
from time import time
from tensorboardX import SummaryWriter
import argparse
import kornia

import torch
from loss import SimpleLoss, DiscriminativeLoss

from data.dataset_front import semantic_dataset
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou
from evaluation.angle_diff import calc_angle_diff
from model_front import get_model
from evaluate import onehot_encoding, eval_iou_2
import random

def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)


def train(args):
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'depth_image_size': args.depth_image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'zgrid': args.zgrid,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }

    train_loader, val_loader = semantic_dataset(
        args.version, args.dataroot, data_conf, args.bsz, args.nworkers, depth_downsample_factor=args.depth_downsample_factor, depth_sup=args.depth_sup, use_depth_enc=args.use_depth_enc, use_depth_enc_bin=args.use_depth_enc_bin, add_depth_channel=args.add_depth_channel,use_lidar_10=True,data_aug=args.data_aug,data_seed=args.data_seed)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim,
                      args.direction_pred, args.angle_class, downsample=args.depth_downsample_factor, use_depth_enc=args.use_depth_enc, pretrained=args.pretrained, add_depth_channel=args.add_depth_channel)

    model.cuda()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, dampening=0.9,
                        weight_decay=args.weight_decay)

    if args.resumef:
        print("Resuming from ", args.resumef)
        checkpoint = torch.load(args.resumef)
        starting_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])

    else:
        print("Training From Scratch ..." )
        starting_epoch = 0

    print("starting_epoch: ", starting_epoch)
    
    writer = SummaryWriter(logdir=args.logdir)

    loss_fn = SimpleLoss(args.pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(
        args.embedding_dim, args.delta_v, args.delta_d).cuda()
    direction_loss_fn = torch.nn.BCELoss(reduction='none')
    depth_loss_func = kornia.losses.FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

    model.train()
    counter = 0
    last_idx = len(train_loader) - 1
    for epoch in range(starting_epoch, args.nepochs):
        model.train()
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans,
                     yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, final_depth_map, final_depth_map_bin_enc, projected_depth) in enumerate(train_loader):
                        
            t0 = time()
            opt.zero_grad()

            semantic, embedding, direction, depth = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                   post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                   lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), final_depth_map_bin_enc.cuda(), projected_depth.cuda())

            semantic_gt = semantic_gt.cuda().float()
            instance_gt = instance_gt.cuda()
            if args.depth_sup:
                final_depth_map = final_depth_map.cuda()
            seg_loss = loss_fn(semantic, semantic_gt)
            if args.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(
                    embedding, instance_gt)
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if args.direction_pred:
                direction_gt = direction_gt.cuda()
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(
                    torch.softmax(direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / \
                    (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(
                    direction, direction_gt, args.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0

            if args.depth_sup:
                depth_loss = depth_loss_func(depth, final_depth_map)
            else:
                depth_loss = 0

            final_loss = seg_loss * args.scale_seg + var_loss * args.scale_var + \
                dist_loss * args.scale_dist + direction_loss * args.scale_direction + depth_loss*args.scale_depth
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                intersects, union = get_batch_iou(
                    onehot_encoding(semantic), semantic_gt)
                iou = intersects / (union + 1e-7)
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1-t0:>7.4f}    "
                            f"Loss: {final_loss.item():>7.4f}    "
                            f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

                write_log(writer, iou, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/direction_loss',
                                  direction_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)
                writer.add_scalar('train/angle_diff', angle_diff, counter)


        iou = eval_iou_2(model, val_loader)
        logger.info(f"EVAL[{epoch:>2d}]:    "
                    f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

        write_log(writer, iou, 'eval', counter)

        model_name = os.path.join(args.logdir, f"model.pt")

        state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict(),
                }
        torch.save(state, model_name)
        logger.info(f"{model_name} saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SuperFusion training.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='/path/to/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='SuperFusion')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
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
    parser.add_argument("--zgrid", nargs=3, type=float,
                        default=[-3.0, 1.5, 0.15])
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

    # depth config
    parser.add_argument('--depth_sup', action='store_true')

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)
    parser.add_argument("--scale_depth", type=float, default=1.0)
    parser.add_argument("--opt", type=str, default='sgd')

    parser.add_argument('--use_depth_enc', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--use_depth_enc_bin', action='store_true')
    parser.add_argument('--add_depth_channel', action='store_true')

    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--data_seed', action='store_true')

    parser.add_argument('--resumef', type=str, default=None)

    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train(args)
