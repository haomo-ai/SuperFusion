import torch
import tqdm

from evaluation.dataset import HDMapNetEvalDataset
from evaluation.chamfer_distance import semantic_mask_chamfer_dist_cum
from evaluation.AP import instance_mask_AP
from evaluation.iou import get_batch_iou

SAMPLED_RECALLS = torch.linspace(0.1, 1, 10)
THRESHOLDS = [0.2, 0.5, 1.0]


def get_val_info(args):
    data_conf = {
        'xbound': args.xbound,
        'ybound': args.ybound,
        'thickness': args.thickness,
    }

    dataset = HDMapNetEvalDataset(
        args.version, args.dataroot, args.eval_set, args.result_path, data_conf)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.bsz, shuffle=False, drop_last=False)

    total_CD1 = torch.zeros(args.max_channel).cuda()
    total_CD2 = torch.zeros(args.max_channel).cuda()
    total_CD_num1 = torch.zeros(args.max_channel).cuda()
    total_CD_num2 = torch.zeros(args.max_channel).cuda()
    total_intersect = torch.zeros(args.max_channel).cuda()
    total_union = torch.zeros(args.max_channel).cuda()
    AP_matrix = torch.zeros((args.max_channel, len(THRESHOLDS))).cuda()
    AP_count_matrix = torch.zeros((args.max_channel, len(THRESHOLDS))).cuda()

    total_CD1_30_60 = torch.zeros(args.max_channel).cuda()
    total_CD2_30_60 = torch.zeros(args.max_channel).cuda()
    total_CD_num1_30_60 = torch.zeros(args.max_channel).cuda()
    total_CD_num2_30_60 = torch.zeros(args.max_channel).cuda()
    total_intersect_30_60 = torch.zeros(args.max_channel).cuda()
    total_union_30_60 = torch.zeros(args.max_channel).cuda()
    AP_matrix_30_60 = torch.zeros((args.max_channel, len(THRESHOLDS))).cuda()
    AP_count_matrix_30_60 = torch.zeros((args.max_channel, len(THRESHOLDS))).cuda()

    total_CD1_60_90 = torch.zeros(args.max_channel).cuda()
    total_CD2_60_90 = torch.zeros(args.max_channel).cuda()
    total_CD_num1_60_90 = torch.zeros(args.max_channel).cuda()
    total_CD_num2_60_90 = torch.zeros(args.max_channel).cuda()
    total_intersect_60_90 = torch.zeros(args.max_channel).cuda()
    total_union_60_90 = torch.zeros(args.max_channel).cuda()
    AP_matrix_60_90 = torch.zeros((args.max_channel, len(THRESHOLDS))).cuda()
    AP_count_matrix_60_90 = torch.zeros((args.max_channel, len(THRESHOLDS))).cuda()

    print('running eval...')
    for pred_map, confidence_level, gt_map in tqdm.tqdm(data_loader):
        # iou
        pred_map = pred_map.cuda() # torch.Size([4, 3, 200, 400])
        confidence_level = confidence_level.cuda()
        gt_map = gt_map.cuda()

        split = int(pred_map.shape[3]/3)

        #intersect, union = get_batch_iou(pred_map[:,:,:,:split], gt_map[:,:,:,:split])
        CD1, CD2, num1, num2 = semantic_mask_chamfer_dist_cum(
            pred_map[:,:,:,:split], gt_map[:,:,:,:split], args.xbound[2], args.ybound[2], threshold=args.CD_threshold)

        instance_mask_AP(AP_matrix, AP_count_matrix, pred_map[:,:,:,:split], gt_map[:,:,:,:split], args.xbound[2], args.ybound[2],
                         confidence_level, THRESHOLDS, sampled_recalls=SAMPLED_RECALLS, bidirectional=args.bidirectional, threshold_iou=args.threshold_iou)

        #total_intersect += intersect.cuda()
        #total_union += union.cuda()
        total_CD1 += CD1
        total_CD2 += CD2
        total_CD_num1 += num1
        total_CD_num2 += num2


        #intersect, union = get_batch_iou(pred_map[:,:,:,split:2*split], gt_map[:,:,:,split:2*split])
        CD1, CD2, num1, num2 = semantic_mask_chamfer_dist_cum(
            pred_map[:,:,:,split:2*split], gt_map[:,:,:,split:2*split], args.xbound[2], args.ybound[2], threshold=args.CD_threshold)

        instance_mask_AP(AP_matrix_30_60, AP_count_matrix_30_60, pred_map[:,:,:,split:2*split], gt_map[:,:,:,split:2*split], args.xbound[2], args.ybound[2],
                         confidence_level, THRESHOLDS, sampled_recalls=SAMPLED_RECALLS, bidirectional=args.bidirectional, threshold_iou=args.threshold_iou)

        #total_intersect_30_60 += intersect.cuda()
        #total_union_30_60 += union.cuda()
        total_CD1_30_60 += CD1
        total_CD2_30_60 += CD2
        total_CD_num1_30_60 += num1
        total_CD_num2_30_60 += num2



        #intersect, union = get_batch_iou(pred_map[:,:,:,2*split:], gt_map[:,:,:,2*split:])
        CD1, CD2, num1, num2 = semantic_mask_chamfer_dist_cum(
            pred_map[:,:,:,2*split:], gt_map[:,:,:,2*split:], args.xbound[2], args.ybound[2], threshold=args.CD_threshold)

        instance_mask_AP(AP_matrix_60_90, AP_count_matrix_60_90, pred_map[:,:,:,2*split:], gt_map[:,:,:,2*split:], args.xbound[2], args.ybound[2],
                         confidence_level, THRESHOLDS, sampled_recalls=SAMPLED_RECALLS, bidirectional=args.bidirectional, threshold_iou=args.threshold_iou)

        #total_intersect_60_90 += intersect.cuda()
        #total_union_60_90 += union.cuda()
        total_CD1_60_90 += CD1
        total_CD2_60_90 += CD2
        total_CD_num1_60_90 += num1
        total_CD_num2_60_90 += num2

    CD_pred_0_30 = total_CD1 / total_CD_num1
    CD_label_0_30 = total_CD2 / total_CD_num2
    CD_0_30 = (total_CD1 + total_CD2) / (total_CD_num1 + total_CD_num2)
    CD_pred_0_30[CD_pred_0_30 > args.CD_threshold] = args.CD_threshold
    CD_label_0_30[CD_label_0_30 > args.CD_threshold] = args.CD_threshold
    CD_0_30[CD_0_30 > args.CD_threshold] = args.CD_threshold


    CD_pred_30_60 = total_CD1_30_60 / total_CD_num1_30_60
    CD_label_30_60 = total_CD2_30_60 / total_CD_num2_30_60
    CD_30_60 = (total_CD1_30_60 + total_CD2_30_60) / (total_CD_num1_30_60 + total_CD_num2_30_60)
    CD_pred_30_60[CD_pred_30_60 > args.CD_threshold] = args.CD_threshold
    CD_label_30_60[CD_label_30_60 > args.CD_threshold] = args.CD_threshold
    CD_30_60[CD_30_60 > args.CD_threshold] = args.CD_threshold


    CD_pred_60_90 = total_CD1_60_90 / total_CD_num1_60_90
    CD_label_60_90 = total_CD2_60_90 / total_CD_num2_60_90
    CD_60_90 = (total_CD1_60_90 + total_CD2_60_90) / (total_CD_num1_60_90 + total_CD_num2_60_90)
    CD_pred_60_90[CD_pred_60_90 > args.CD_threshold] = args.CD_threshold
    CD_label_60_90[CD_label_60_90 > args.CD_threshold] = args.CD_threshold
    CD_60_90[CD_60_90 > args.CD_threshold] = args.CD_threshold

    CD_pred = (total_CD1 + total_CD1_30_60 + total_CD1_60_90) / (total_CD_num1 + total_CD_num1_30_60 + total_CD_num1_60_90) 
    CD_label = (total_CD2 + total_CD2_30_60 + total_CD2_60_90) / (total_CD_num2 + total_CD_num2_30_60 + total_CD_num2_60_90) 
    CD = (total_CD1 + total_CD2+ total_CD1_30_60 + total_CD2_30_60+ total_CD1_60_90+ total_CD2_60_90) / (total_CD_num1 +total_CD_num2 +  total_CD_num1_30_60+  total_CD_num2_30_60 + total_CD_num1_60_90+ total_CD_num2_60_90) 
    AP = (AP_matrix+AP_matrix_30_60+AP_matrix_60_90) / (AP_count_matrix+AP_count_matrix_30_60+AP_count_matrix_60_90)
    
    return {
        #'iou_0_30': total_intersect / total_union,
        'CD_pred_0_30': CD_pred_0_30,
        'CD_label_0_30': CD_label_0_30,
        'CD_0_30': CD_0_30,
        'Average_precision_0_30': AP_matrix / AP_count_matrix,
        #'iou_30_60': total_intersect_30_60 / total_union_30_60,
        'CD_pred_30_60': CD_pred_30_60,
        'CD_label_30_60': CD_label_30_60,
        'CD_30_60': CD_30_60,
        'Average_precision_30_60': AP_matrix_30_60 / AP_count_matrix_30_60,
        #'iou_60_90': total_intersect_60_90 / total_union_60_90,
        'CD_pred_60_90': CD_pred_60_90,
        'CD_label_60_90': CD_label_60_90,
        'CD_60_90': CD_60_90,
        'Average_precision_60_90': AP_matrix_60_90 / AP_count_matrix_60_90,
        'CD_pred': CD_pred,
        'CD_label': CD_label,
        'CD': CD,
        'AP': AP,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate nuScenes local HD Map Construction Results.')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--dataroot', type=str,
                        default='/path/to/nuScenes/')
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--eval_set', type=str, default='val',
                        choices=['train', 'val', 'test', 'mini_train', 'mini_val'])
    parser.add_argument('--thickness', type=int, default=5)
    parser.add_argument('--max_channel', type=int, default=3)
    parser.add_argument('--CD_threshold', type=int, default=5)
    parser.add_argument("--xbound", nargs=3, type=float,
                        default=[-90.0, 90.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float,
                        default=[-15.0, 15.0, 0.15])
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--threshold_iou', type=float, default=0.1)

    args = parser.parse_args()

    print(get_val_info(args))
