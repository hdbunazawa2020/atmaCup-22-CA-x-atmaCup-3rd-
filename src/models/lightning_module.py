import pytorch_lightning as pl
import torch
import torch.nn as nn
from timm.utils import ModelEmaV3
from torchmetrics.classification import MulticlassF1Score
import sys
sys.path.append("..")
from models.fusion_model import PlayerFusionEmbeddingModel
from training.step_functions import train_fn, valid_fn


class PlayerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        num_tabular_features: int,
        embedding_dim: int,
        pretrained: bool,
        arcface_s: float,
        arcface_m: float,
        lr: float,
        weight_decay: float,
        epochs: int,
        use_ema: bool = True,
        ema_decay: float = 0.995,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = PlayerFusionEmbeddingModel(
            model_name=model_name,
            num_classes=num_classes,
            num_tabular_features=num_tabular_features,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arcface_s=arcface_s,
            arcface_m=arcface_m,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.model_ema = None

    def setup(self, stage: str | None = None):
        if self.use_ema and self.model_ema is None:
            self.model_ema = ModelEmaV3(self.model, decay=self.ema_decay, device=self.device)

    def on_before_zero_grad(self, optimizer):
        # optimizer.step() の後に呼ばれるのでEMA更新に丁度いい
        if self.use_ema and self.model_ema is not None:
            self.model_ema.update(self.model)

    def training_step(self, batch, batch_idx):
        loss, preds, y = train_fn(self, batch)
        self.train_f1(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = valid_fn(self, batch)
        self.val_f1(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def on_save_checkpoint(self, checkpoint):
        # ckpt肥大化を避けたいならEMA重みを落とす（必要ならコメントアウト）
        if "state_dict" in checkpoint:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith("model_ema."):
                    del checkpoint["state_dict"][k]