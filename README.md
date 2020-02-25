# RetinaFace in PyTorch

A [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). The official code in Mxnet can be found [here](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

## TODO

[ ] Move to detectron2

## WiderFace Val Performance in single scale When using Mobilenet0.25 as backbone net.

| Style                       |  easy   | medium |  hard  |
|:----------------------------|:-------:|:------:|:------:|
| Mxnet(Single Scale 2150)    | 88.72%  | 86.97% | 79.19% |
| Mxnet(Original Scale)       | 89.58%  | 87.11% | 69.12% |
| Pytorch(Same as Mxnet 2150) | 91.795% | 90.49% | 85.55% |
| Pytorch(Original Scale)     | 92.65%  | 90.91% | 80.23% |

## Dependencies

* pytorch == 1.3.1
* torchvision == 0.4.0
* python == 3.7

## Installation

```shell
cd libs/DCNv2
./make.sh
```

## Run Demo

```shell
python3 demo.py
```
