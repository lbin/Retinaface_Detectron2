import math
from typing import List

import torch
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.meta_arch import RetinaNet
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

# from aifwdet.layers.nms import batched_nms
from aifwdet.layers.ssh import SSH

__all__ = ["RetinaFace"]


@META_ARCH_REGISTRY.register()
class RetinaFace(RetinaNet):
    """
    # Implement RetinaFace (https://arxiv.org/abs/1905.00641).
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = RetinaFaceHead(cfg, feature_shapes)
        self.input_format = cfg.INPUT.FORMAT
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.input_format == "RGB":
            images = [(x / 255.0 - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class RetinaFaceHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels      = input_shape[0].channels
        num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs        = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        self.ssh = []
        for i in range(len(cfg.MODEL.RETINANET.IN_FEATURES)):
            ssh = SSH(cfg, in_channels, in_channels)
            name = "ssh" + str(i)
            self.add_module(name, ssh)
            self.ssh.append(ssh)

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())

        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=1, stride=1, padding=0
        )
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1, padding=0)

        # Initialization
        for modules in [self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for i, feature in enumerate(features):
            feature = self.ssh[i](feature)
            logits.append(self.cls_score(feature))
            bbox_reg.append(self.bbox_pred(feature))
        return logits, bbox_reg
