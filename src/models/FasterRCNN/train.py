
# # import logging
# # import torch
# # import hydra
# # from hydra.utils import instantiate 
# # from omegaconf import DictConfig, OmegaConf
# # from pytorch_lightning import Trainer
# # from pytorch_lightning.callbacks import ModelCheckpoint
# # from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
# # from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# # from .dataset import DotaDataModule
# # from .model import DotaLightningModel

# # logger = logging.getLogger(__name__)

# # @hydra.main(config_path="configs", config_name="config", version_base="1.3")
# # def train(cfg: DictConfig):
# #     logger.info("🔧 Training Config:")
# #     logger.info(OmegaConf.to_yaml(cfg))

# #     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
# #     # 1) instantiate the pretrained Faster R-CNN
# #     weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# #     model = fasterrcnn_resnet50_fpn_v2(weights=weights)

# #     # ─── 2) FREEZE THE BACKBONE ──────────────────────────────────────────
# #     for param in model.backbone.body.parameters():
# #         param.requires_grad = False
# #     for param in model.backbone.fpn.parameters():
# #         param.requires_grad = False
# #     for m in model.backbone.modules():
# #         if isinstance(m, torch.nn.BatchNorm2d):
# #             m.eval()
# #     # ───────────────────────────────────────────────────────────────────

# #     # 3) swap in a fresh predictor head (its params remain requires_grad=True)
# #     in_features = model.roi_heads.box_predictor.cls_score.in_features
# #     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.data.num_classes)

# #     # 4) build optimizer over all trainable params (no groups)
# #     optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
# #     # instantiate optimizer on all parameters that require grad
# #     optimizer = instantiate(optimizer_cfg, 
# #                             params=[p for p in model.parameters() if p.requires_grad],
# #                             _convert_="partial")
# #     scheduler = instantiate(cfg.scheduler, optimizer=optimizer, _convert_="partial")

# #     model = DotaLightningModel(model=model, optimizer=optimizer, scheduler=scheduler)

# #     datamodule = DotaDataModule(
# #         batch_size=cfg.training.batch_size,
# #         num_workers=cfg.training.num_workers,
# #         transforms=None
# #     )

# #     checkpoint_callback = ModelCheckpoint(
# #         dirpath=cfg.training.output_dir,
# #         filename="dota-model-{epoch:02d}-{map_50:.3f}",
# #         save_top_k=1,
# #         verbose=True,
# #         monitor="map_50",
# #         mode="max"
# #     )

# #     trainer = Trainer(
# #         max_epochs=cfg.training.epochs,
# #         accelerator="gpu" if torch.cuda.is_available() else "cpu",
# #         devices=1,
# #         log_every_n_steps=cfg.training.log_every_n_steps,
# #         default_root_dir=cfg.training.output_dir,
# #         precision=cfg.training.precision,
# #         callbacks=[checkpoint_callback],
# #         overfit_batches=2,
# #     )

# #     trainer.fit(
# #         model,
# #         datamodule=datamodule
# #     )

# # if __name__ == "__main__":
# #     train()
# import logging
# import torch
# import hydra
# from hydra.utils import instantiate 
# from omegaconf import DictConfig, OmegaConf
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint
# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# from .dataset import DotaDataModule
# from .model import DotaLightningModel

# logger = logging.getLogger(__name__)

# @hydra.main(config_path="configs", config_name="config", version_base="1.3")
# def train(cfg: DictConfig):
#     logger.info("🔧 Training Config:")
#     logger.info(OmegaConf.to_yaml(cfg))

#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
#     # 1) instantiate the pretrained Faster R-CNN with COCO weights
#     weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
#     model = fasterrcnn_resnet50_fpn_v2(weights=weights)

#     # ─── 2) FREEZE THE BACKBONE ──────────────────────────────────────────
#     for param in model.backbone.body.parameters():
#         param.requires_grad = False
#     for param in model.backbone.fpn.parameters():
#         param.requires_grad = False
#     for m in model.backbone.modules():
#         if isinstance(m, torch.nn.BatchNorm2d):
#             m.eval()
#     # ───────────────────────────────────────────────────────────────────

#     # 3) swap in a fresh predictor head (its params remain requires_grad=True)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.data.num_classes)

#     # ─── 3.5) LOWER DETECTION THRESHOLDS ─────────────────────────────────
#     # reduce minimum score for a detection to be retained
#     model.roi_heads.score_thresh       = cfg.model.detection_threshold  # e.g. 0.01
#     # adjust NMS threshold if desired
#     model.roi_heads.nms_thresh         = cfg.model.nms_threshold        # e.g. 0.7
#     # allow more detections per image
#     model.roi_heads.detections_per_img = cfg.model.max_detections_per_img  # e.g. 200
#     # ───────────────────────────────────────────────────────────────────

#     # 4) build optimizer over all trainable params (no groups)
#     optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
#     optimizer = instantiate(
#         optimizer_cfg, 
#         params=[p for p in model.parameters() if p.requires_grad],
#         _convert_="partial"
#     )

#     # 5) build and attach scheduler
#     scheduler = instantiate(
#         cfg.scheduler,
#         optimizer=optimizer,
#         _convert_="partial"
#     )

#     model = DotaLightningModel(model=model, optimizer=optimizer, scheduler=scheduler)

#     datamodule = DotaDataModule(
#         batch_size=cfg.training.batch_size,
#         num_workers=cfg.training.num_workers,
#         transforms=None
#     )

#     checkpoint_callback = ModelCheckpoint(
#         dirpath=cfg.training.output_dir,
#         filename="dota-model-{epoch:02d}-{map_50:.3f}",
#         save_top_k=1,
#         verbose=True,
#         monitor="map_50",
#         mode="max"
#     )

#     trainer = Trainer(
#         max_epochs=cfg.training.epochs,
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#         devices=1,
#         log_every_n_steps=cfg.training.log_every_n_steps,
#         default_root_dir=cfg.training.output_dir,
#         precision=cfg.training.precision,
#         callbacks=[checkpoint_callback],
#         overfit_batches=2,
#     )

#     trainer.fit(
#         model,
#         datamodule=datamodule
#     )

# if __name__ == "__main__":
#     train()
#!/usr/bin/env python3
"""
Minimal script to fine‑tune Faster R‑CNN on DOTA with a frozen backbone for the first 10 epochs,
then unfreeze the backbone for fine‑tuning the entire network.
"""
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from models.FasterRCNN.dataset import DotaDataModule, DotaDataset  # adjust import path


class UnfreezeBackboneCallback(Callback):
    """
    Callback to unfreeze the backbone at a specified epoch.
    """
    def __init__(self, unfreeze_epoch: int = 10):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            # unfreeze all backbone parameters
            for param in pl_module.model.backbone.parameters():
                param.requires_grad = True
            pl_module.print(f"⚙️ Unfroze backbone at epoch {self.unfreeze_epoch}")


def main():
    # 1) DataModule
    dm = DotaDataModule(
        batch_size=4,
        num_workers=4,
        transforms=None  # model’s own transform handles normalization & resize
    )
    dm.setup(stage="fit")  # initialize datasets

    # 2) Compute number of classes
    raw_labels = []
    for label_path in dm.train_dataset.labels:
        labels, _ = DotaDataset.read_labels(label_path)
        raw_labels.extend(labels)
    num_real = max(raw_labels) + 1  # labels are 0-based
    num_classes = num_real + 1      # +1 for background

    # 3) Build model with pretrained COCO weights
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )

    # 4) Freeze backbone for initial warm-up
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 5) Replace the predictor head
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

    # 6) Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    scheduler = StepLR(optimizer, step_size=33, gamma=0.1)

    # 7) LightningModule
    from models.FasterRCNN.model import DotaLightningModel
    lit_model = DotaLightningModel(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # attach scheduler via override if needed, else DotaLightningModel.configure_optimizers uses self.scheduler
    def configure_optimizers_override(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
    lit_model.configure_optimizers = configure_optimizers_override.__get__(lit_model, DotaLightningModel)

    # 8) Callbacks: checkpoint + unfreeze
    ckpt_cb = ModelCheckpoint(
        save_top_k=1,
        monitor="map_50",
        mode="max",
        filename="dota-finetune-{epoch:02d}-{map_50:.3f}"
    )
    unfreeze_cb = UnfreezeBackboneCallback(unfreeze_epoch=10)

    # 9) Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        callbacks=[ckpt_cb, unfreeze_cb],
        log_every_n_steps=10,
    )

    # 10) Run training
    trainer.fit(lit_model, dm)


if __name__ == "__main__":
    main()
