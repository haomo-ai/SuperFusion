from torch import nn
from typing import Any, Optional
from torchvision.models._utils import IntermediateLayerGetter
#from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models import resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
import torch

__all__ = ['deeplabv3_resnet101']


model_urls = {
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}


def _segm_model(
    name: str,
    backbone_name: str,
    num_classes: int,
    use_depth_enc: bool,
    aux: Optional[bool],
    pretrained_backbone: bool = True
) -> nn.Module:
    if 'resnet' in backbone_name:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True])
        out_layer = 'layer4'
        if use_depth_enc:
            out_inplanes = 2048 + 128
        else:
            out_inplanes = 2048
        aux_layer = 'layer3'
        aux_inplanes = 1024
    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    return_layers = {out_layer: 'out'}
    if aux:
        return_layers[aux_layer] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # if aux:
    #     aux_classifier = FCNHead(aux_inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
    }
    classifier = model_map[name][0](out_inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(
    arch_type: str,
    backbone: str,
    pretrained: bool,
    progress: bool,
    num_classes: int,
    use_depth_enc: bool,
    aux_loss: Optional[bool],
    **kwargs: Any
) -> nn.Module:
    if pretrained:
        aux_loss = True
        kwargs["pretrained_backbone"] = False
    model = _segm_model(arch_type, backbone, num_classes, use_depth_enc, aux_loss, **kwargs)
    if pretrained:
        _load_weights(model, arch_type, backbone, progress)
    return model


def _load_weights(model: nn.Module, arch_type: str, backbone: str, progress: bool) -> None:
    arch = arch_type + '_' + backbone + '_coco'
    model_url = model_urls.get(arch, None)
    if model_url is None:
        raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
    else:
        state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict)


def deeplabv3_resnet101(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    use_depth_enc = False,
    **kwargs: Any
) -> nn.Module:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, use_depth_enc, aux_loss, **kwargs)


