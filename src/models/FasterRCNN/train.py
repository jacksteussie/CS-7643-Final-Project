import logging
import torch
import hydra
from hydra.utils import instantiate 
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .dataset import DotaDataModule
from .model import DotaLightningModel

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    
    logger.info("🔧 Training Config:")
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 1) instantiate the pretrained Faster R-CNN
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # ─── 2) FREEZE THE BACKBONE ────────────────────────────────────────────
    # freeze all ResNet layers:
    for param in model.backbone.body.parameters():
        param.requires_grad = False
    # freeze all FPN layers:
    for param in model.backbone.fpn.parameters():
        param.requires_grad = False
    # (optional) set frozen BatchNorms to eval mode so they use running stats:
    for m in model.backbone.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    # ──────────────────────────────────────────────────────────────────────

    # 3) swap in a fresh predictor head (its params remain requires_grad=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.data.num_classes)

    # 4) build optimizer param_groups by only grabbing .requires_grad==True
    param_groups = []
    for group_cfg in cfg.optimizer.param_groups:
        name = group_cfg.name
        if name == "backbone":
            # this will now be empty, since we froze it
            params = [p for _, p in model.backbone.named_parameters() if p.requires_grad]
        elif name == "head":
            params = [p for _, p in model.roi_heads.named_parameters() if p.requires_grad]
        else:
            raise ValueError(f"Unknown param group name: {name}")

        group_cfg_clean = OmegaConf.to_container(group_cfg, resolve=True)
        group_cfg_clean.pop("name")
        group_cfg_clean["params"] = params
        param_groups.append(group_cfg_clean)

    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    optimizer_cfg.pop("param_groups", None)
    optimizer = instantiate(optimizer_cfg, param_groups, _convert_="partial")

    model = DotaLightningModel(
        model=model, 
        optimizer=optimizer
    )

    datamodule = DotaDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        transforms=None  # or replace with actual transforms
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.output_dir,
        filename="dota-model-{epoch:02d}-{map_50:.3f}",
        save_top_k=1,
        verbose=True,
        monitor="map_50",    # ← now watches your validation mAP@.50
        mode="max"           # ← higher map_50 is better
    )

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=cfg.training.log_every_n_steps,
        default_root_dir=cfg.training.output_dir,
        precision=cfg.training.precision,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.training.get("resume_from_checkpoint", None)
    )

if __name__ == "__main__":
    train()
