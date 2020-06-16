import os

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode


def get_widerface_metadata():
    metadata = {"thing_classes": ["face"]}
    return metadata


def get_widerface_dicts(image_root):
    label_file = os.path.join(image_root, "label.txt")

    imgs_path = []
    imgs_path_no_head = []
    words = []

    with open(label_file) as f:
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith("#"):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                imgs_path_no_head.append(path)
                path = label_file.replace("label.txt", "images/") + path
                imgs_path.append(path)
            else:
                line = line.split(" ")
                label = [float(x) for x in line]
                labels.append(label)

        words.append(labels)

    widerface_dicts = []
    for index in range(len(words)):

        filename = imgs_path[index]
        height, width = cv2.imread(filename).shape[:2]

        record = {}
        record["file_name"] = filename
        record["image_id"] = imgs_path_no_head[index]
        record["height"] = height
        record["width"] = width

        labels = words[index]
        # annotations = np.zeros((0, 15))
        objs = []

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1

            if label[0] >= width or label[1] >= height:
                continue

            if label[2] <= 0 or label[3] <= 0:
                continue

            annotation[0, 2] = label[0] + label[2]  # x2
            if annotation[0, 2] >= width:
                annotation[0, 2] = width - 1

            annotation[0, 3] = label[1] + label[3]  # y2
            if annotation[0, 3] >= height:
                annotation[0, 3] = height - 1

            if len(label) > 4:
                # landmarks
                annotation[0, 4] = label[4]  # l0_x
                annotation[0, 5] = label[5]  # l0_y
                annotation[0, 6] = label[7]  # l1_x
                annotation[0, 7] = label[8]  # l1_y
                annotation[0, 8] = label[10]  # l2_x
                annotation[0, 9] = label[11]  # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
                if annotation[0, 4] < 0:
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1
            obj = {
                "bbox": [annotation[0, 0], annotation[0, 1], annotation[0, 2], annotation[0, 3]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "landmark": annotation,
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        widerface_dicts.append(record)
    return widerface_dicts


def register_widerface():
    SPLITS = {
        "widerface_train": ("widerface/train", "widerface/train/label.txt"),
        "widerface_val": ("widerface/val", "widerface/val/label.txt"),
    }
    for name, (image_root, label_file) in SPLITS.items():
        label_file = os.path.join("datasets", label_file)
        image_root = os.path.join("datasets", image_root)
        register_widerface_instance(name, image_root)


def register_widerface_instance(name, image_root):
    DatasetCatalog.register(name, lambda name=name: get_widerface_dicts(image_root))
    MetadataCatalog.get(name).set(**get_widerface_metadata())
