import logging
import os
from collections import OrderedDict, defaultdict

import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils import comm

from .widerface_evaluation import evaluation


class WiderFaceEvaluator(DatasetEvaluator):
    """
    Evaluate Wider Face AP.
    It contains a synchronization, therefore has to be called from all ranks.
    """

    def __init__(self, dataset_name, output_folder):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "widerface_val"
        """
        self._dataset_name = dataset_name
        self._output_folder = output_folder
        meta = MetadataCatalog.get(dataset_name)
        # data_info = DatasetCatalog.get(dataset_name)

        self._class_names = meta.thing_classes

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1

                self._predictions[image_id].append([xmin, ymin, xmax, ymax, score])

            if len(self._predictions[image_id]) == 0:
                self._predictions[image_id].append([0, 0, 0, 0, 0])

    def evaluate(self):

        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        tmp_results_path = os.path.join(self._output_folder, "wider_face_val_results")

        for image_id in predictions.keys():
            tmp_results_file = tmp_results_path + "/" + image_id[:-4] + ".txt"
            dirname = os.path.dirname(tmp_results_file)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)

            with open(tmp_results_file, "w") as fd:
                # bboxs = dets
                file_name = os.path.basename(tmp_results_file)[:-4] + "\n"
                bboxs_num = str(len(predictions[image_id])) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                idx = 0
                for box in predictions[image_id]:

                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(float(box[4]))
                    line = (
                        str(x)
                        + " "
                        + str(y)
                        + " "
                        + str(w)
                        + " "
                        + str(h)
                        + " "
                        + confidence
                        + " \n"
                    )
                    fd.write(line)
                    idx = idx + 1

        aps = evaluation(tmp_results_path, "datasets/widerface/val/ground_truth")

        ret = OrderedDict()
        ret["bbox"] = {"Easy": aps[0], "Medium": aps[1], "Hard": aps[2]}
        return ret
