import numpy as np
from PIL import Image

import torch
import torchvision


class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))

normalize_img_aug = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.ColorJitter(
               brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))

normalize_img_aug_flip = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(p=1.0),
    torchvision.transforms.ColorJitter(
               brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))

denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))


def img_transform(img, resize, resize_dims):
    post_rot2 = torch.eye(2)
    post_tran2 = torch.zeros(2)

    img = img.resize(resize_dims)

    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot2 = rot_resize @ post_rot2
    post_tran2 = rot_resize @ post_tran2

    post_tran = torch.zeros(3)
    post_rot = torch.eye(3)
    post_tran[:2] = post_tran2
    post_rot[:2, :2] = post_rot2
    return img, post_rot, post_tran


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

