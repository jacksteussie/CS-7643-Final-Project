import torch
import torch.nn as nn
from ultralytics.utils.loss import RotatedBboxLoss

from utils import obb_to_xywha

class OBBLossWrapper(nn.Module):
    def __init__(self, num_classes, device="cuda", reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.bbox_loss = RotatedBboxLoss(reg_max=reg_max).to(device)

    def forward(self, cls_logits, box_deltas, angle_preds, targets, proposals):
        """
        cls_logits: [B, N, C]
        box_deltas: [B, N, 4] (x1, y1, x2, y2)
        angle_preds: [B, N, 1]
        targets: List[Dict] with 'boxes' [N, 8] and 'labels' [N]
        proposals: same size as preds (needed for anchor_points)
        """
        B, N, _ = cls_logits.shape

        total_cls_loss = 0.0
        total_box_loss = 0.0
        total_dfl_loss = 0.0

        for i in range(B):
            # Class Loss
            labels = targets[i]['labels'].to(self.device)
            total_cls_loss += self.ce(cls_logits[i], labels)

            # Convert ground truth to cx, cy, w, h, angle
            gt_boxes = obb_to_xywha(targets[i]['boxes'].to(self.device))
            
            # Convert predicted boxes (x1y1x2y2) to cx, cy, w, h
            pred_boxes = box_deltas[i]
            x1, y1, x2, y2 = pred_boxes.unbind(-1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = torch.abs(x2 - x1)
            h = torch.abs(y2 - y1)
            angle = angle_preds[i].squeeze(-1)

            pred_rot = torch.stack([cx, cy, w, h, angle], dim=-1)

            fg_mask = torch.ones(len(gt_boxes), dtype=torch.bool, device=self.device)
            fake_scores = torch.ones(len(gt_boxes), dtype=torch.float32, device=self.device)

            box_loss, dfl_loss = self.bbox_loss(
                pred_dist=None,  # Skip DFL if not used
                pred=pred_rot,
                anchor_points=proposals[i][:, :2].to(self.device),  # dummy anchor points
                target=gt_boxes,
                target_scores=fake_scores,
                target_scores_sum=len(gt_boxes),
                fg_mask=fg_mask
            )

            total_box_loss += box_loss
            total_dfl_loss += dfl_loss

        return total_cls_loss, total_box_loss, total_dfl_loss
