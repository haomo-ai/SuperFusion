import numpy as np
import math
import torch


def get_proj_mat(intrins, rots, trans):
    K = np.eye(4)
    K[:3, :3] = intrins
    R = np.eye(4)
    R[:3, :3] = rots.transpose(-1, -2)
    T = np.eye(4)
    T[:3, 3] = -trans
    RT = R @ T
    return K @ RT


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def label_onehot_decoding(onehot):
    return torch.argmax(onehot, axis=0)


def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1)
    return onehot

def ego_to_cam_points(points, rot, trans):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    return points

def project_pc_to_image(point_cloud, cam_p):
    """Projects a 3D point cloud to 2D points

    Args:
        point_cloud: (3, N) point cloud
        cam_p: camera projection matrix

    Returns:
        pts_2d: (2, N) projected coordinates [u, v] of the 3D points
    """
    points = cam_p.matmul(point_cloud)
    points[:2] /= points[2:3]

    return points[0:2]

def project_depths(point_cloud, cam_p, image_shape, max_depth=100.0):
    """Projects a point cloud into image space and saves depths per pixel.

    Args:
        point_cloud: (3, N) Point cloud in cam0
        cam_p: camera projection matrix
        image_shape: image shape [h, w]
        max_depth: optional, max depth for inversion

    Returns:
        projected_depths: projected depth map
    """

    # Only keep points in front of the camera
    all_points = point_cloud.T

    # Save the depth corresponding to each point
    points_in_img = project_pc_to_image(all_points.T, cam_p)
    points_in_img_int = np.int32(np.round(points_in_img))

    # Remove points outside image
    valid_indices = \
        (points_in_img_int[0] >= 0) & (points_in_img_int[0] < image_shape[1]) & \
        (points_in_img_int[1] >= 0) & (points_in_img_int[1] < image_shape[0])

    all_points = all_points[valid_indices]
    points_in_img_int = points_in_img_int[:, valid_indices]

    # Invert depths
    all_points[:, 2] = max_depth - all_points[:, 2]

    # Only save valid pixels, keep closer points when overlapping
    projected_depths = np.zeros(image_shape)
    valid_indices = [points_in_img_int[1], points_in_img_int[0]]
    projected_depths[tuple(valid_indices)] = [
        max(projected_depths[
            points_in_img_int[1, idx], points_in_img_int[0, idx]],
            all_points[idx, 2])
        for idx in range(points_in_img_int.shape[1])]

    projected_depths[tuple(valid_indices)] = \
        max_depth - projected_depths[tuple(valid_indices)]

    projected_depths[projected_depths < 0] = 0

    return projected_depths.astype(np.float32)

def bin_depths(depth_map, depth_min, depth_max, bin_size, target=True, mode="UD", add_depth_enc=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        num_bins = (depth_max - depth_min) / bin_size
        indices = ((depth_map - depth_min) / bin_size)
    else:
        num_bins = (depth_max - depth_min) / bin_size
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * np.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~np.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = torch.Tensor(indices).type(torch.int64)

    if add_depth_enc:
        # Remove indicies outside of bounds
        mask = (indices > num_bins) | (~np.isfinite(indices))
        indices[mask] = num_bins

        mask = (indices < 0)
        indices[mask] = -1

        # Convert to integer
        indices = torch.Tensor(indices).type(torch.int64)
    return indices

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                          for row in [xbound, ybound, zbound]])
    return dx, bx, nx


# dx, bx, nx = gen_dx_bx(
#     [-30.0, 30.0, 0.15], [-15.0, 15.0, 0.15], [-10.0, 10.0, 20.0])
# print(dx)
# print(bx)
# print(nx)
