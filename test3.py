def postprocess(self, preds, img, orig_imgs):
    print("postprocess")  # 添加这行来检查方法是否被调用
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

            # Convert to [x_center, y_center, width, height]
            bbox_xywh[:, 0] = bbox_xywh[:, 0] + bbox_xywh[:, 2] / 2
            bbox_xywh[:, 1] = bbox_xywh[:, 1] + bbox_xywh[:, 3] / 2

            outputs = self.deepsort.update(bbox_xywh, confs, cls, orig_img)

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]

                self.draw_boxes(orig_img, bbox_xyxy, self.model.names, object_id, identities)

        results.append(pred)
    return results