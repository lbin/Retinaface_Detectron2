# from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C

# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #

_C.MODEL.RETINANET.WITH_DCNv2 = False
_C.MODEL.RETINANET.NORM = "BN"
