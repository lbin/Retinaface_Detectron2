# RetinaFace in PyTorch

A [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). The official code in Mxnet can be found [here](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

Old version canbe found at [v1.0](https://github.com/lbin/Retinaface_Mobilenet_Pytorch/tree/v1.0)


## WiderFace Val Performance in single scale When using ResNet50 as backbone net.

| Style                 |  easy  | medium |  hard  |
| :-------------------- | :----: | :----: | :----: |
| Ours (Original Scale) | 94.14% | 92.71% | 81.13% |

## Dependencies

* pytorch >= 1.4.0
* torchvision >= 0.4.0
* python >= 3.6

## Installation

pip install -e .
