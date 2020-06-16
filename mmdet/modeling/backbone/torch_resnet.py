import torch.nn as nn
import torchvision.models.resnet as resnet
from detectron2.layers import ShapeSpec

# from centernet.network.backbone import Backbone
from detectron2.modeling import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

_resnet_mapper = {18: resnet.resnet18, 50: resnet.resnet50, 101: resnet.resnet101}


class ResnetBackbone(Backbone):
    def __init__(self, cfg, out_features=None, pretrained=True):
        super().__init__()
        depth = cfg.MODEL.RESNETS.DEPTH
        backbone = _resnet_mapper[depth](pretrained=pretrained)
        self.stage0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

        self.stages_and_names = []

        self.add_module("res1", self.stage0)
        self.stages_and_names.append((self.stage0, "res1"))

        self.add_module("res2", self.stage1)
        self.stages_and_names.append((self.stage1, "res2"))

        self.add_module("res3", self.stage2)
        self.stages_and_names.append((self.stage2, "res3"))

        self.add_module("res4", self.stage3)
        self.stages_and_names.append((self.stage3, "res4"))

        self.add_module("res5", self.stage4)
        self.stages_and_names.append((self.stage4, "res5"))

        self._out_feature_strides = {}
        self._out_feature_channels = {}

        self._out_feature_strides["res3"] = 8
        self._out_feature_channels["res3"] = 512

        self._out_feature_strides["res4"] = 16
        self._out_feature_channels["res4"] = 1024

        self._out_feature_strides["res5"] = 32
        self._out_feature_channels["res5"] = 2048

        self._out_features = out_features

    def forward(self, x):
        outputs = {}
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x

        return outputs


@BACKBONE_REGISTRY.register()
def build_torch_resnet_backbone(cfg):
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    return ResnetBackbone(cfg, out_features)


@BACKBONE_REGISTRY.register()
def build_torch_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_torch_resnet_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
