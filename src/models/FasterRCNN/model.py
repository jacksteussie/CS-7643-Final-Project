import pytorch_lightning as pl
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import logging

logger = logging.getLogger(__name__)


class DotaLightningModel(pl.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self._optimizer = optimizer
        self.metric = MeanAveragePrecision(iou_thresholds=[0.5])

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        batch_size = len(images)

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = len(batch[0])
        optimizer = self.trainer.optimizers[0]

        if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
            lr = optimizer.param_groups[0]['lr']
            self.log(
                'lr',
                lr,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size
            )

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        self.metric.update(outputs, targets)

    def on_validation_epoch_end(self):
        metrics = self.metric.compute()
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                self.log(k, v.item(), prog_bar=True, logger=True)
            elif isinstance(v, (float, int)):
                self.log(k, v, prog_bar=True, logger=True)
        self.metric.reset()

    def configure_optimizers(self):
        return self._optimizer
