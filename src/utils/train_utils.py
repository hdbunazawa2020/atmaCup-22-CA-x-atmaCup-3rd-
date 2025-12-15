import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR

def get_criterion(cfg, mode="valid"):
    """損失関数の取得"""
    if mode == "train":
        if cfg.criterion == "l1":
            return nn.L1Loss()
        elif cfg.criterion == "mse":
            return nn.MSELoss()
        elif cfg.criterion == "huber":
            return nn.HuberLoss(delta=1.0) # SmoothL1Loss
        else:
            raise ValueError(f"Unsupported loss function: {cfg.criterion}")
    else:
        return nn.L1Loss() # MAE

def get_optimizer(cfg, model):
    """オプティマイザの取得"""
    if cfg.optimizer.name == "adam":
        return Adam(model.parameters(), lr=cfg.optimizer.lr)
    elif cfg.optimizer.name == "adamw":
        return AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, fused=True)
    elif cfg.optimizer.name == "sgd":
        return SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

class ConstantCosineLR(_LRScheduler):
    """
    前半は一定、後半はCosineAnnealingする学習率スケジューラ。
    """
    def __init__(self, optimizer, total_steps, pct_cosine=0.5, last_epoch=-1):
        self.total_steps = total_steps
        self.milestone = int(total_steps * (1 - pct_cosine))
        self.cosine_steps = max(self.total_steps - self.milestone, 1)
        self.min_lr = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.milestone:
            factor = 1.0
        else:
            s = step - self.milestone
            factor = 0.5 * (1 + math.cos(math.pi * s / self.cosine_steps))
        return [base_lr * factor for base_lr in self.base_lrs]

class WarmupCosineLR(LambdaLR):
    def __init__(self, optimizer, total_steps, warmup_steps, base_lr, max_lr, min_lr):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return self.base_lr + (self.max_lr - self.base_lr) * (current_step / warmup_steps)
            else:
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
                return lr / self.max_lr  # LambdaLR expects ratio w.r.t initial lr

        super().__init__(optimizer, lr_lambda)

def get_scheduler(cfg, optimizer):
    """スケジューラの取得"""
    name = cfg.scheduler.name
    if name == "none":
        return None
    elif name == "cosine":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.scheduler.T_0,
            T_mult=cfg.scheduler.T_mult,
            eta_min=cfg.scheduler.eta_min,
        )
    elif name == "constantcosine":
        return ConstantCosineLR(
            optimizer,
            total_steps=cfg.scheduler.total_steps,
            pct_cosine=cfg.scheduler.pct_cosine,
        )
    elif name == "warmup_cosine":
        return WarmupCosineLR(
            optimizer,
            total_steps=cfg.scheduler.total_steps,
            warmup_steps=cfg.scheduler.warmup_steps,
            base_lr=cfg.scheduler.base_lr,
            max_lr=cfg.scheduler.max_lr,
            min_lr=cfg.scheduler.min_lr,
        )
    elif name == "step":
        return StepLR(
            optimizer,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
        )
    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

def get_scaler(cfg):
    """AMP用GradScalerの取得（cfgに応じて切り替え可能）"""
    return GradScaler(enabled=getattr(cfg, "use_amp", True))

def gradient_loss(pred, target):
    """
    境界のエッジ差分にペナルティを与えるGradient Loss
    pred, target: (B, 1, H, W) or (B, H, W)
    """
    # 次元調整（必要なら）
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    # x方向
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    # y方向
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_dx = torch.abs(pred_dx - target_dx).mean()
    loss_dy = torch.abs(pred_dy - target_dy).mean()

    return loss_dx + loss_dy