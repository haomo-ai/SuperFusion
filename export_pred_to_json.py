import argparse
import mmcv
import tqdm
import torch

from data.dataset_front import semantic_dataset
from data.const import NUM_CLASSES
from model_front import get_model
from postprocess.vectorize import vectorize


def gen_dx_bx(xbound, ybound):
    dx = [row[2] for row in [xbound, ybound]]
    bx = [row[0] + row[2] / 2.0 for row in [xbound, ybound]]
    nx = [(row[1] - row[0]) / row[2] for row in [xbound, ybound]]
    return dx, bx, nx


def export_to_json(model, val_loader, angle_class, args):
    submission = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_external": False,
            "vector": True,
        },
        "results": {}
    }

    dx, bx, nx = gen_dx_bx(args.xbound, args.ybound)

    model.eval()
    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt, final_depth_map, final_depth_map_bin_enc, projected_depth) in enumerate(tqdm.tqdm(val_loader)):
            # if args.model == 'HDMapNet_fusion':
            #     segmentation, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
            #                                             post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
            #                                             lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            # else:
            segmentation, embedding, direction, _ = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                    post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                    lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), final_depth_map_bin_enc.cuda(), projected_depth.cuda())

            for si in range(segmentation.shape[0]):
                coords, confidences, line_types = vectorize(
                    segmentation[si], embedding[si], direction[si], angle_class)
                vectors = []
                for coord, confidence, line_type in zip(coords, confidences, line_types):
                    vector = {'pts': coord * dx + bx, 'pts_num': len(
                        coord), "type": line_type, "confidence_level": confidence}
                    vectors.append(vector)
                rec = val_loader.dataset.samples[batchi *
                                                 val_loader.batch_size + si]
                submission['results'][rec['token']] = vectors

    mmcv.dump(submission, args.output)


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
    model = get_model(args.model, data_conf, True, args.embedding_dim,
        True, args.angle_class, downsample=args.depth_downsample_factor, use_depth_enc=args.use_depth_enc, pretrained=args.pretrained, add_depth_channel=args.add_depth_channel,add_fuser=args.add_fuser)

    if args.model == 'HDMapNet_fusion' or args.model == 'HDMapNet_cam':
        model.load_state_dict(torch.load(args.modelf), strict=False)
    else:
        checkpoint = torch.load(args.modelf)
        model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    if args.eval_set == 'val':
        export_to_json(model, val_loader, args.angle_class, args)
    else:
        export_to_json(model, train_loader, args.angle_class, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # nuScenes config
    parser.add_argument('--dataroot', type=str,
                        default='/path/to/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='SuperFusion')

    parser.add_argument('--eval_set', type=str, default='val',
                        choices=['train', 'val'])
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--train_loader_shuffle', action='store_true')

    # training config
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)

    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[256, 704])
    parser.add_argument("--xbound", nargs=3, type=float,
                        default=[0.0, 90.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float,
                        default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float,
                        default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float,
                        default=[2.0, 90.0, 1.0])

    # embedding config
    parser.add_argument("--embedding_dim", type=int, default=16)

    # direction config
    parser.add_argument('--angle_class', type=int, default=36)

    # output
    parser.add_argument("--output", type=str, default='output.json')
    parser.add_argument('--lidar_cut_x', action='store_true')
    parser.add_argument("--camC", type=int, default=64)
    parser.add_argument("--lidarC", type=int, default=128)
    parser.add_argument("--crossC", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument('--cross_atten', action='store_true')
    parser.add_argument('--cross_conv', action='store_true')
    parser.add_argument('--add_bn', action='store_true')


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
