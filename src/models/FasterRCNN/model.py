import pytorch_lightning as pl
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import logging

logger = logging.getLogger(__name__)

class DotaLightningModel(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = MeanAveragePrecision(iou_thresholds=[0.5])

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        self.metric.update(outputs, targets)

    def on_validation_epoch_end(self):
        metrics = self.metric.compute()

        # Only log scalar-compatible metrics
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    self.log(k, v.item(), prog_bar=True, logger=True)
                else:
                    logger.debug(f"Skipping non-scalar metric: {k} = {v}")
            elif isinstance(v, (float, int)):
                self.log(k, v, prog_bar=True)
            else:
                logger.debug(f"Skipping unknown-type metric: {k} = {v}")

        self.metric.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]

