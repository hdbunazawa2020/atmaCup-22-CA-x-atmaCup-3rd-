import os
from pathlib import Path

import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import WandbLogger

import sys
sys.path.append("..")
from utils.data import save_config_yaml, dict_to_namespace
from utils.wandb_utils import set_wandb

from atma_datasets.datamodule import PlayerDataModule
from atma_datasets.transforms import get_val_transform
from atma_datasets.player_dataset import PlayerDataset
from models.lightning_module import PlayerLightningModule

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")

import torch
from torch.utils.data import DataLoader


# ===================================
# utils
# ===================================
@torch.no_grad()
def extract_val_embeddings(module, val_loader, device):
    module.eval()
    embs = []
    ys = []
    for batch in val_loader:
        img = batch["image"].to(device)
        tab = batch["num"].to(device)
        y = batch["label"].cpu().numpy()
        # EMAがあるならEMAでembedding
        if module.use_ema and module.model_ema is not None:
            e = module.model_ema.module.get_embedding(img, tab).detach().cpu().numpy()
        else:
            e = module.model.get_embedding(img, tab).detach().cpu().numpy()
        embs.append(e)
        ys.append(y)
    return np.concatenate(embs), np.concatenate(ys)

@torch.no_grad()
def compute_prototypes(module, loader, device, num_classes: int):
    module.eval()
    embs = []
    ys = []
    for batch in loader:
        img = batch["image"].to(device)
        tab = batch["num"].to(device)
        y = batch["label"].cpu().numpy()

        if module.use_ema and module.model_ema is not None:
            e = module.model_ema.module.get_embedding(img, tab).detach().cpu().numpy()
        else:
            e = module.model.get_embedding(img, tab).detach().cpu().numpy()

        embs.append(e); ys.append(y)

    embs = np.concatenate(embs, axis=0)  # (N, D)
    ys = np.concatenate(ys, axis=0)      # (N,)

    # クラスごと平均 → 正規化
    D = embs.shape[1]
    prot = np.zeros((num_classes, D), dtype=np.float32)
    for c in range(num_classes):
        m = (ys == c)
        if m.any():
            v = embs[m].mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-8)
            prot[c] = v
    return prot

# ===================================
# main
# ===================================
# TODO: config_pathをこのスクリプトからの相対パスにする
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """description
    Args:
        cfg (DictConf): config
    """
    # set config
    config_dict = OmegaConf.to_container(cfg["100_train_arcface"], resolve=True)
    config = dict_to_namespace(config_dict)
    pl.seed_everything(config.seed, workers=True)
    # when debug
    if config.debug:
        config.exp = "100_train_arcface_debug" # TODO: ファイルの連番を入れる
    # make savedir
    savedir = Path(config.output_dir) / config.exp
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir / "oof", exist_ok=True)
    os.makedirs(savedir / "yaml", exist_ok=True)
    os.makedirs(savedir / "model", exist_ok=True)
    
    # wandb
    wandb_logger = None
    if config.use_wandb:
        set_wandb(config)  # あなたの utils 側に寄せる
        wandb_logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name,
            save_dir=str(savedir),
            log_model=True,
        )
        wandb_logger.log_hyperparams(config_dict)

    # ---- datamodule ----
    num_features = list(config.num_features) if hasattr(config, "num_features") else []
    dm = PlayerDataModule(
        train_pp_csv=Path(config.train_pp_csv),
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_features=num_features,
        split_method=getattr(config, "split_method", "holdout"),
        val_quarter_from=getattr(config, "val_quarter_from", "Q2-016"),
        n_splits=getattr(config, "n_splits", 5),
        fold=getattr(config, "fold", 0),
        seed=config.seed,
        p_hflip=getattr(config, "p_hflip", 0.5),
    )

    # ---- module ----
    module = PlayerLightningModule(
        model_name=config.model_name,
        num_classes=config.num_classes,
        num_tabular_features=len(num_features) if getattr(config, "use_tabular", True) else 0,
        embedding_dim=config.embedding_dim,
        pretrained=config.pretrained,
        arcface_s=config.arcface_s,
        arcface_m=config.arcface_m,
        lr=config.lr,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        use_ema=config.use_ema,
        ema_decay=config.ema_decay,
    )

    # ---- callbacks ----
    model_dir = Path(savedir) / "model"  # または Path(config.model_dir)
    ckpt = ModelCheckpoint(
        dirpath=str(model_dir),
        filename="best",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
        enable_version_counter=False,
    )

    pbar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="cyan",
            progress_bar="blue",
            progress_bar_finished="bright_blue",
            progress_bar_pulse="#0080FF",
            batch_progress="cyan",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )

    callbacks = [ckpt, pbar]
    if config.use_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=wandb_logger if config.use_wandb else False,
        precision="16-mixed",
        deterministic=True,
        default_root_dir=str(savedir),
    )
    trainer.fit(module, dm)
    print("best ckpt:", ckpt.best_model_path)

    # ---- OOF & prototypes (no augmentation loaders) ----
    val_tf = get_val_transform(config.img_size)

    train_ds_noaug = PlayerDataset(
        df=dm.train_df,
        transform=val_tf,
        num_features=num_features,
        is_train=False,
        p_hflip=0.0,
    )
    val_ds_noaug = PlayerDataset(
        df=dm.val_df,
        transform=val_tf,
        num_features=num_features,
        is_train=False,
        p_hflip=0.0,
    )

    train_loader = DataLoader(train_ds_noaug, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds_noaug, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    device = module.device

    # embeddings
    emb, y = extract_val_embeddings(module, val_loader, device)

    oof_dir = Path(savedir) / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    np.save(oof_dir / "val_emb.npy", emb)
    np.save(oof_dir / "val_y.npy", y)

    # prototypes
    prototypes = compute_prototypes(module, train_loader, device, num_classes=config.num_classes)

    scores = emb @ prototypes.T
    pred = scores.argmax(axis=1)

    oof_df = pd.DataFrame({
        "y": y,
        "pred": pred,
        "max_sim": scores.max(axis=1),
    })
    oof_df.to_csv(oof_dir / "oof_df.csv", index=False)
    print("saved oof:", oof_dir)

if __name__ == "__main__":
    main()