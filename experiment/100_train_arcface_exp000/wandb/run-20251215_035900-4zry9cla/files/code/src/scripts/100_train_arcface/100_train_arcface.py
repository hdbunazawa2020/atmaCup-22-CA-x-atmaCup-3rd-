import os
from pathlib import Path

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
from models.lightning_module import PlayerLightningModule

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")

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
    ckpt = ModelCheckpoint(
        dirpath=str(savedir),
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


if __name__ == "__main__":
    main()