import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FastRCNNDFLPredictor(FastRCNNPredictor):
    """Subclass in order to use faster-r-cnn with DFL loss provided by ultralytics"""
    def __init__(self, in_channels, num_classes, reg_max=16):
        super().__init__(in_channels, num_classes)
        # The original head does not provide an additional "distribution head" which is necessary for DFL
        self.reg_max = reg_max
        self.dfl_pred = nn.Linear(in_channels, 4 * reg_max)
        self.angle_pred = nn.Linear(in_channels, 1)

    def forward(self, x):
        # x: [#proposals, in_channels] or [N, C, 1, 1]
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        scores = self.cls_score(x)          # → [N, num_classes]
        bbox_deltas = self.bbox_pred(x)     # → [N, 4]
        dfl_logits = self.dfl_pred(x)       # → [N, 4*reg_max]
        angle_logits = self.angle_pred(x)   # → [N, 1]
        return scores, bbox_deltas, dfl_logits, angle_logits
