import torchvision

from .ddn_template import DDNTemplate
from .segmentation import deeplabv3_resnet101

class DDNDeepLabV3(DDNTemplate):

    def __init__(self, backbone_name, **kwargs):
        """
        Initializes DDNDeepLabV3 model
        Args:
            backbone_name [str]: ResNet Backbone Name
        """
        if backbone_name == "ResNet101":
            constructor = deeplabv3_resnet101
        else:
            raise NotImplementedError

        super().__init__(constructor=constructor, **kwargs)
