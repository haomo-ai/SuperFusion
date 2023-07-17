import argparse
import tqdm

import torch

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def eval_iou(model, val_loader):
    model.eval()
    total_intersects = 0
    total_union = 0
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt in tqdm.tqdm(val_loader):

            semantic, embedding, direction, img_feat, img_feat_inv = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                                           post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                                           lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())

            semantic_gt = semantic_gt.cuda().float()
            intersects, union = get_batch_iou(
                onehot_encoding(semantic), semantic_gt)
            total_intersects += intersects
            total_union += union
    return total_intersects / (total_union + 1e-7)


def eval_iou_2(model, val_loader):
    model.eval()
    total_intersects = 0
    total_union = 0
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, final_depth_map, final_depth_map_bin, projected_depth in tqdm.tqdm(val_loader):

            semantic, embedding, direction, depth = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), final_depth_map_bin.cuda(), projected_depth.cuda())


            semantic_gt = semantic_gt.cuda().float()
            intersects, union = get_batch_iou(
                onehot_encoding(semantic), semantic_gt)
            total_intersects += intersects
            total_union += union
    return total_intersects / (total_union + 1e-7)

