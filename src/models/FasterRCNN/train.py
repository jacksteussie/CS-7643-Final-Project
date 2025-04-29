import logging
import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    EarlyStopping,
    Callback,
)
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .dataset import DotaDataModule, DotaDataset
from .model import DotaLightningModel

# ─── Setup Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class UnfreezeBackboneCallback(Callback):
    """
    Callback to unfreeze the backbone at a specified epoch.
    """
    def __init__(self, unfreeze_epoch: int = 10):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            logger.info(f"⚙️ Unfreezing backbone at epoch {self.unfreeze_epoch}")
            for param in pl_module.model.backbone.parameters():
                param.requires_grad = True

@hydra.main(config_path="configs", config_name="config2", version_base="1.3")
def main(cfg: DictConfig):
    logger.info("🔧 Config:\n" + OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision("medium")
    logger.info("ℹ️ Set float32 matmul precision to 'medium'")

    # ─── Data Module ───────────────────────────────────────────────────────────
    dm = DotaDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    dm.setup(stage="fit")

    # ─── Determine num_classes ──────────────────────────────────────────────────
    
    num_classes = cfg.data.get("num_classes")
    if not num_classes:
        try:
            train_dataset = dm.train_dataloader().dataset
            raw_labels = []
            for lp in getattr(train_dataset, 'labels', []):
                labels, _ = DotaDataset.read_labels(lp)
                raw_labels.extend(labels)
            max_label = max(raw_labels)
            num_classes = max_label + 2  # classes + background
        except Exception:
            logger.error("Could not determine num_classes; please set cfg.data.num_classes.")
            return
    logger.info(f"✅ Using num_classes={num_classes}")

    resume_ckpt: str | None = cfg.training.get("resume_from_checkpoint", None)
    if resume_ckpt:
        logger.info(f"🔄 Resuming training from checkpoint: {resume_ckpt}")     

    # ─── Build and prepare model ─────────────────────────────────────────────────
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    for param in model.backbone.parameters():
        param.requires_grad = False
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    logger.info("✅ Model built and backbone frozen.")

    # ─── Optimizer Setup ─────────────────────────────────────────────────────────
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
    )
    logger.info("✅ Optimizer (SGD) initialized.")

    # ─── LightningModule and Trainer ────────────────────────────────────────────
    lit_model = DotaLightningModel(
        model=model,
        optimizer=optimizer,
    )

    # Pull this out at the top of your main()
    monitor_metric = cfg.training.get("monitor_metric") or "map_50"

    ckpt = ModelCheckpoint(
        save_top_k=cfg.training.get("save_top_k", 1),
        monitor=monitor_metric,
        mode=cfg.training.get("monitor_mode", "max"),
        dirpath=cfg.training.get("save_dir", "./checkpoints"),
        # Note the double braces to escape in an f-string:
        filename = f"dota-{{epoch:02d}}-{{{monitor_metric}:.3f}}",
    )
    callbacks = [
        ckpt,
        RichProgressBar(),
        UnfreezeBackboneCallback(cfg.training.get("unfreeze_epoch", 50)),
    ]
    if cfg.training.get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor=cfg.training.get("monitor_metric", "map_50"),
                patience=cfg.training.get("early_stopping_patience", 5),
                mode=cfg.training.get("monitor_mode", "max"),
            )
        )

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.get("accelerator", "auto"),
        devices=cfg.training.get("devices", "auto"),
        strategy=cfg.training.get("strategy", "auto"),
        callbacks=callbacks,
        precision=cfg.training.get("precision", "16-mixed"),
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.get("val_check_interval", 1.0),
        overfit_batches=cfg.training.get("overfit_batches", 20),
    )

    trainer.fit(lit_model, datamodule=dm, ckpt_path=resume_ckpt)
    logger.info("✅ Training complete.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
