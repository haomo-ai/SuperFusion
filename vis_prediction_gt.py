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



def vis_vector(model, val_loader, angle_class):
    # model.eval()
    car_img = Image.open('pics/car.png')
    colors_plt = ['r', 'b', 'g']

    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt, final_depth_map_bin, final_depth_map_bin_enc, projected_depth, vectors,rec) in enumerate(val_loader):
            for si in range(1):

                plt.figure(figsize=(4, 2))
                plt.xlim(0, 90)
                plt.ylim(-15, 15)
                plt.axis('off')

                for vector in vectors:
                    pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
                    pts = pts[:pts_num].cpu().detach().numpy()
                    pts = pts[0, :]
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.plot(x, y, color=colors_plt[line_type])

                plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])

                print("rec: ", rec['data']['CAM_FRONT'])
                map_path = 'results/'+args.saveroot + \
                    f'/eval_{batchi:04}_'+str(rec['data']['CAM_FRONT'])+'_gt.jpg'
                print('saving', map_path)
                plt.savefig(map_path, bbox_inches='tight', dpi=400)
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
        args.version, args.dataroot, data_conf, args.bsz, args.nworkers, depth_downsample_factor=args.depth_downsample_factor, depth_sup=args.depth_sup, use_depth_enc=args.use_depth_enc, use_depth_enc_bin=args.use_depth_enc_bin, add_depth_channel=args.add_depth_channel,use_lidar_10=args.use_lidar_10, visual=True)

    model = None
    vis_vector(model, val_loader, args.angle_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # nuScenes config
    parser.add_argument('--dataroot', type=str,
                        default='/media/hao/HaoData/dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
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

    parser.add_argument('--saveroot', type=str,
                        default='SuperFusion')

    parser.add_argument('--depth_sup', action='store_true')
    parser.add_argument('--use_depth_enc', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--use_depth_enc_bin', action='store_true')
    parser.add_argument('--add_depth_channel', action='store_true')
    parser.add_argument('--use_lidar_10', action='store_true')


    args = parser.parse_args()
    main(args)
