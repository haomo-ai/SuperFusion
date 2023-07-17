import torch
from torch import nn
import math

from .pointpillar import PointPillarEncoder
from .base import BevEncode
from data.utils import gen_dx_bx

from .ddn_deeplabv3 import DDNDeepLabV3
import torch.nn.functional as F
from inplace_abn import InPlaceABNSync

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Initializes convolutional block for channel reduce
        Args:
            out_channels [int]: Number of output channels of convolutional block
            **kwargs [Dict]: Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features [torch.Tensor(B, C_in, H, W)]: Input features
        Returns:
            x [torch.Tensor(B, C_out, H, W)]: Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CamEncode(nn.Module):
    def __init__(self, D, C, use_depth_enc=False, pretrained=True, add_depth_channel=False):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.use_depth_enc = use_depth_enc
        self.add_depth_channel = add_depth_channel

        if pretrained:
            print("use pretrain")
            self.ddn = DDNDeepLabV3(
                num_classes=self.D + 1,
                backbone_name="ResNet101",
                feat_extract_layer="layer1",
                pretrained_path="checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth",
                use_depth_enc = use_depth_enc,
                add_depth_channel = add_depth_channel
            )
        else:
            print("no pretrain")
            self.ddn = DDNDeepLabV3(
                num_classes=self.D + 1,
                backbone_name="ResNet101",
                feat_extract_layer="layer1",
                #pretrained_path="checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth",
                use_depth_enc = use_depth_enc,
                add_depth_channel = add_depth_channel
            )
        
        self.channel_reduce = BasicBlock2D(256, 64)
        if self.use_depth_enc:
            self.depth_enc = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x, depth, projected_depth):
        if self.use_depth_enc:
            encoded_depth = self.depth_enc(depth)
            ddn_result = self.ddn(x, encoded_depth)
        elif self.add_depth_channel:
            ddn_result = self.ddn(x, projected_depth=projected_depth)
        else:
            ddn_result = self.ddn(x)
        image_features = ddn_result["features"]
        depth_logits = ddn_result["logits"]
        image_features_out = ddn_result["image_features_out"]
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)

        frustum_features = self.create_frustum_features(image_features=image_features,
                                                        depth_logits=depth_logits)

        return depth_logits, frustum_features, image_features_out

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth classification scores
        Args:
            image_features [torch.Tensor(N, C, H, W)]: Image features
            depth_logits [torch.Tensor(N, D, H, W)]: Depth classification logits
        Returns:
            frustum_features [torch.Tensor(N, C, D, H, W)]: Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features
        return frustum_features

    def forward(self, x, depth_enc, projected_depth):
        depth, x, image_features_out = self.get_depth_feat(x, depth_enc, projected_depth)

        return x, depth, image_features_out


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, camC=64, downsample=16, use_depth_enc=False, pretrained=True, add_depth_channel=False):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf

        self.grid_conf['xbound'] = [
            0.0, self.grid_conf['xbound'][1], self.grid_conf['xbound'][2]]

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = downsample
        self.camC = camC
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, use_depth_enc=use_depth_enc, pretrained=pretrained, add_depth_channel=add_depth_channel)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.grid_conf['image_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x, depth_enc, projected_depth):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x, depth, image_features_out = self.camencode(x, depth_enc, projected_depth)
        x = x.view(B, N, self.camC, self.D, imH //
                   self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, depth, image_features_out

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros(
            (B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2],
              geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans, depth_enc, projected_depth):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x, depth, image_features_out = self.get_cam_feats(x, depth_enc, projected_depth)

        x = self.voxel_pooling(geom, x)

        return x, depth, image_features_out

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, depth_enc, projected_depth):
        x, depth, image_features_out = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans, depth_enc, projected_depth)
        return x.transpose(3, 2), depth, image_features_out

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :,
        :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class LidarPred(nn.Module):
    def __init__(self, use_cross=False, num_heads=1, pos_emd=True, neck_dim=256, cross_dim=256):
        super(LidarPred, self).__init__()
        self.use_cross = use_cross
        self.pos_emd = pos_emd
        self.conv11 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(256)

        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(256)

        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(256)

        self.conv41d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256)

        self.conv21d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(256)

        self.conv11d = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)


        if self.use_cross:
            self.conv_reduce = nn.Conv2d(2048, neck_dim, kernel_size=1)
            if pos_emd:
                self.pe_lidar = positionalencoding2d(
                    neck_dim, 25*2, 75*2).cuda()
                self.pe_img = positionalencoding2d(
                    neck_dim, 32, 88).cuda()
            self.multihead_attn = nn.MultiheadAttention(
                    neck_dim, num_heads, dropout=0.3, batch_first=True)
            self.conv = nn.Sequential(
                        nn.Conv2d(neck_dim+cross_dim,
                                  neck_dim, kernel_size=3, padding=1),
                        nn.BatchNorm2d(neck_dim),
                        nn.ReLU(inplace=True)
                    )
            self.conv_cross = nn.Sequential(
                nn.Conv2d(neck_dim,
                            cross_dim, kernel_size=1),
                nn.BatchNorm2d(cross_dim),
                nn.ReLU(inplace=True)
            )
            
    def cross_attention(self, x, img_feature):
        B, C, H, W = x.shape
        if self.pos_emd:
            pe_lidar = self.pe_lidar.repeat(B, 1, 1, 1)
            pe_img = self.pe_img.repeat(B, 1, 1, 1)
            x = x + pe_lidar
            img_feature = img_feature + pe_img
        query = x.reshape(B, C, -1).permute(0, 2, 1)
        key = img_feature.reshape(B, C, -1).permute(0, 2, 1)
        attn_output, attn_output_weights = self.multihead_attn(
            query, key, key)
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)
        attn_output = self.conv_cross(attn_output)
        fused_feature = torch.cat([x, attn_output],
                                    dim=1)
        fused_feature = self.conv(fused_feature)
        return fused_feature

    def forward(self, x, img_feature=None):
        x11 = F.relu(self.bn11(self.conv11(x)))
        x1p, id1 = self.max_pool(x11)

        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x2p, id2 = self.max_pool(x21)

        x41 = F.relu(self.bn41(self.conv41(x2p))) # bottleneck

        if self.use_cross:
            img_feature = self.conv_reduce(img_feature)
            x41 = self.cross_attention(x41, img_feature)

        x41d = F.relu(self.bn41d(self.conv41d(x41)))

        x3d = self.max_unpool(x41d, id2)
        x21d = F.relu(self.bn21d(self.conv21d(x3d)))

        x2d = self.max_unpool(x21d, id1)

        x11d = self.conv11d(x2d)

        return x11d


class AlignFAnew(nn.Module):
    def __init__(self, features):
        super(AlignFAnew, self).__init__()

        self.delta_gen1 = nn.Sequential(
                        nn.Conv2d(features, 128, kernel_size=1, bias=False),
                        InPlaceABNSync(128),
                        nn.Conv2d(128, 2, kernel_size=3, padding=1, bias=False)
                        )


        self.delta_gen1[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w-1)/s, (out_h-1)/s]]]]).type_as(input).to(input.device) 
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        
        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        low_stage = self.bilinear_interpolate_torch_gridsample2(low_stage, (h, w), delta1)

        concat = torch.cat((low_stage, high_stage), 1)
        return concat


class SuperFusion(nn.Module):
    def __init__(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False, camC=64, lidarC=128, downsample=16, use_depth_enc=False, pretrained=True, add_depth_channel=False, ppdim=15):
        super(SuperFusion, self).__init__()
        self.camC = camC
        self.lidarC = lidarC
        self.downsample = downsample
        self.add_depth_channel = add_depth_channel


        self.lss = LiftSplatShoot(data_conf, camC, downsample = downsample, use_depth_enc=use_depth_enc, pretrained=pretrained, add_depth_channel=add_depth_channel)

        self.lidar = lidar

        self.fuser_AlignFA = AlignFAnew(self.camC+self.lidarC)

        self.lidar_pred = LidarPred(use_cross=True, num_heads=4, neck_dim=256, cross_dim=256)

        if lidar:
            self.pp = PointPillarEncoder(
                self.lidarC, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'], ppdim=ppdim)
            self.bevencode = BevEncode(inC=self.camC+self.lidarC, outC=data_conf['num_channels'], instance_seg=instance_seg,
                                       embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], instance_seg=instance_seg,
                                       embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, depth_enc, projected_depth):
        # torch.Size([4, 6, 64, 8, 22])
        topdown, depth, image_features_out = self.lss(img, rots, trans, intrins, post_rots, post_trans, depth_enc, projected_depth)
        # print(topdown.shape)

        if self.lidar:
            lidar_feature, neck_feature = self.pp(
                lidar_data, lidar_mask)  
            lidar_feature = self.lidar_pred(lidar_feature, image_features_out)

            topdown = self.fuser_AlignFA(topdown, lidar_feature)
                                
        x, x_embedded, x_direction = self.bevencode(topdown)
        return x, x_embedded, x_direction, depth
