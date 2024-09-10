from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import select_device

import torch
import numpy as np
from pathlib import Path

# DeepSORT imports
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque


class DeepSORTPredictor(BasePredictor):

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.deepsort = None
        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        self.data_deque = {}

    def setup_model(self, model, device):
        super().setup_model(model, device)
        self.init_tracker()

    def init_tracker(self):
        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

        self.deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                                 max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                                 min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg_deep.DEEPSORT.MAX_AGE,
                                 n_init=cfg_deep.DEEPSORT.N_INIT,
                                 nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            # DeepSORT tracking
            if pred is not None and len(pred):
                bbox_xywh = pred[:, :4].cpu()
                confs = pred[:, 4].cpu()
                cls = pred[:, 5].cpu()

                bbox_xywh[:, 2:] -= bbox_xywh[:, :2]  # Convert to xywh format
                outputs = self.deepsort.update(bbox_xywh, confs, cls, orig_img)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]

                    self.draw_boxes(orig_img, bbox_xyxy, self.model.names, object_id, identities)

            results.append(Results(orig_img=orig_img, path=self.batch[0], names=self.model.names, boxes=pred))
        return results

    def draw_boxes(self, img, bbox, names, object_id, identities=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            # code to find center of bottom edge
            center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

            # get ID of object
            id = int(identities[i]) if identities is not None else 0

            # create new buffer for new object
            if id not in self.data_deque:
                self.data_deque[id] = deque(maxlen=64)

            color = self.compute_color_for_labels(object_id[i])
            obj_name = names[int(object_id[i])]
            label = f'{id}:{obj_name}'

            # add center to buffer
            self.data_deque[id].appendleft(center)
            self.ui_box(img, box, label=label, color=color, line_thickness=2)

            # draw trail
            for i in range(1, len(self.data_deque[id])):
                # check if on buffer value is none
                if self.data_deque[id][i - 1] is None or self.data_deque[id][i] is None:
                    continue
                # generate dynamic thickness of trails
                thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                # draw trails
                cv2.line(img, self.data_deque[id][i - 1], self.data_deque[id][i], color, thickness)

    def compute_color_for_labels(self, label):
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)

    def ui_box(self, img, box, label='', color=(128, 128, 128), line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def predict(cfg=None, use_python=False):
    model = YOLO(cfg)
    model.predictor = DeepSORTPredictor(overrides=dict(conf=0.25, iou=0.45, agnostic_nms=False, max_det=300))
    return model(source=source, stream=stream)


if __name__ == '__main__':
    predict()