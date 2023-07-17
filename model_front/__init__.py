from .super_fusion import SuperFusion

def get_model(method, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36, camC=64, lidarC=128, crossC=128, num_heads=1, cross_atten=False, cross_conv=False, add_bn=False, pos_emd=False, pos_emd_img=False, lidar_feature_trans=False, add_fuser=False, downsample=16, use_depth_enc=False, pretrained=True, add_depth_channel=False, ppdim=15):
    if method == 'SuperFusion':
        model = SuperFusion(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim,
                            direction_pred=direction_pred, direction_dim=angle_class, lidar=True, camC=camC, lidarC=lidarC, downsample=downsample, use_depth_enc=use_depth_enc, pretrained=pretrained, add_depth_channel=add_depth_channel, ppdim=ppdim)
    else:
        raise NotImplementedError

    return model
