import torch
from torchvision.ops.boxes import box_convert     # still need box_convert
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import rotated_nms


class FastRCNNDFLPredictor(FastRCNNPredictor):
    """
    Custom head for Faster R-CNN that outputs:
      - classification scores
      - bounding box deltas (xyxy)
      - distribution logits (for DFL)
      - angle prediction
    """
    def __init__(self, in_channels, num_classes, reg_max=16):
        super().__init__(in_channels, num_classes)
        self.reg_max = reg_max
        self.dfl_pred = nn.Linear(in_channels, 4 * reg_max)
        self.angle_pred = nn.Linear(in_channels, 1)

    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        scores       = self.cls_score(x)
        bbox_deltas  = self.bbox_pred(x)
        dfl_logits   = self.dfl_pred(x)
        angle_logits = self.angle_pred(x).squeeze(-1)
        return scores, bbox_deltas, dfl_logits, angle_logits


class RotatedRoIHeads(RoIHeads):
    """
    Modified RoIHeads that decodes DFL + angle to (cx, cy, w, h, theta)
    and uses rotated NMS (pure Python via shapely).
    """
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        reg_max=16,
    ):
        super().__init__(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=fg_iou_thresh,
            bg_iou_thresh=bg_iou_thresh,
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )

        self.reg_max = reg_max
        self.register_buffer("proj", torch.linspace(0, reg_max - 1, reg_max))

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for t in targets:
                # Validate expected keys and types
                assert "boxes" in t and "labels" in t
                assert t["boxes"].dtype == torch.float32
                assert t["labels"].dtype == torch.int64

        # Extract features
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, bbox_deltas, dfl_logits, angle_preds = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, bbox_deltas, proposals, targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                # Add DFL + angle losses here if needed
            }
        else:
            result = self.postprocess_detections(
                class_logits, (bbox_deltas, dfl_logits, angle_preds), proposals, image_shapes
            )

        return result, losses

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        bbox_deltas, dfl_logits, angle_logits = box_regression
        device = bbox_deltas.device

        flat_props = torch.cat(proposals, dim=0)
        scores = class_logits.softmax(-1)
        label_scores, labels = scores[:, 1:].max(-1)

        P = dfl_logits.shape[0]
        dfl = dfl_logits.view(P, 4, self.reg_max).softmax(-1)
        offsets = dfl.matmul(self.proj.to(device))
        dx, dy, dw, dh = offsets.unbind(-1)

        props_cxcywh = box_convert(flat_props, "xyxy", "cxcywh")
        cx, cy, w, h = props_cxcywh.unbind(-1)
        pred_cx = cx + dx * w
        pred_cy = cy + dy * h
        pred_w  = w  * dw.exp()
        pred_h  = h  * dh.exp()
        pred_angle = angle_logits

        rboxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h, pred_angle], dim=1)

        keep = label_scores > self.score_thresh
        kept_inds = keep.nonzero(as_tuple=False).squeeze(1)
        rboxes, label_scores, labels = (
            rboxes[kept_inds], label_scores[kept_inds], labels[kept_inds]
        )

        results = []
        start = 0
        for props in proposals:
            num = props.size(0)
            end = start + num
            mask = (kept_inds >= start) & (kept_inds < end)
            if not mask.any():
                results.append({
                    "boxes":  torch.empty((0, 5), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=device),
                })
            else:
                sel = mask.nonzero(as_tuple=False).squeeze(1)
                boxes_i = rboxes[sel]
                scores_i = label_scores[sel]
                labels_i = labels[sel]
                keep_i = rotated_nms(boxes_i, scores_i, self.nms_thresh)
                results.append({
                    "boxes":  boxes_i[keep_i],
                    "scores": scores_i[keep_i],
                    "labels": labels_i[keep_i],
                })
            start = end

        return results