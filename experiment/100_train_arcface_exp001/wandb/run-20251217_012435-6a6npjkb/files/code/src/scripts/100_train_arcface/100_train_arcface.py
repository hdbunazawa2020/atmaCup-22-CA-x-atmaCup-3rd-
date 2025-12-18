import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")


# ===================================
# utils
# ===================================
def _resolve_pp_paths(config) -> tuple[Path, Path]:
    """
    train_pp_csv / test_pp_csv が config にあればそれを使う。
    無ければ pp_dir / pp_exp から自動生成する。
    """
    if hasattr(config, "train_pp_csv") and str(config.train_pp_csv):
        train_pp = Path(config.train_pp_csv)
    else:
        train_pp = Path(config.pp_dir) / config.pp_exp / "train_meta_pp.csv"

    if hasattr(config, "test_pp_csv") and str(config.test_pp_csv):
        test_pp = Path(config.test_pp_csv)
    else:
        test_pp = Path(config.pp_dir) / config.pp_exp / "test_meta_pp.csv"

    return train_pp, test_pp


def _maybe_drop_junk_and_save(train_pp_csv: Path, savedir: Path, junk_col: str = "is_junk", drop_junk: bool = True) -> Path:
    """
    train_pp_csv を読み、is_junk==1 を落とした CSV を savedir に吐く。
    （DataModule にはこの clean CSV を渡す）
    """
    if not drop_junk:
        return train_pp_csv

    df = pd.read_csv(train_pp_csv)
    if junk_col not in df.columns:
        print(f"[drop_junk] '{junk_col}' not in columns. skip.")
        return train_pp_csv

    before = len(df)
    # is_junk が NaN の場合もあるので 0 扱い
    m_keep = (df[junk_col].fillna(0).astype(int) == 0)
    df2 = df.loc[m_keep].copy()
    after = len(df2)

    out_csv = Path(savedir) / "train_meta_pp_clean.csv"
    df2.to_csv(out_csv, index=False)

    print(f"[drop_junk] {before} -> {after} (dropped {before-after}) saved: {out_csv}")
    # 参考: crop_variant があれば分布も出す
    if "crop_variant" in df.columns:
        print("[drop_junk] crop_variant (before):\n", df["crop_variant"].value_counts())
    if "crop_variant" in df2.columns:
        print("[drop_junk] crop_variant (after):\n", df2["crop_variant"].value_counts())
    return out_csv


def flip_num_feats_batch(x: torch.Tensor, n_players: torch.Tensor, col2idx: dict[str, int]) -> torch.Tensor:
    """
    推論(300)と同様の “数値特徴の左右反転補正” の最低限版。
    x: (B,F) float
    n_players: (B,) long
    """
    x = x.clone()

    # formation PCA x
    if "fx" in col2idx:
        x[:, col2idx["fx"]] = -x[:, col2idx["fx"]]

    # rank_x: n-1-rank_x
    if "rank_x" in col2idx:
        rx = x[:, col2idx["rank_x"]]
        x[:, col2idx["rank_x"]] = (n_players.float() - 1.0) - rx

    # rank_x_norm: 1 - rank_x_norm (n<=1 のときは 0)
    if "rank_x_norm" in col2idx:
        rxn = x[:, col2idx["rank_x_norm"]]
        denom = (n_players.float() - 1.0)
        x[:, col2idx["rank_x_norm"]] = torch.where(denom > 0, 1.0 - rxn, torch.zeros_like(rxn))

    return x


@torch.no_grad()
def extract_embeddings(
    module: PlayerLightningModule,
    loader: DataLoader,
    device: torch.device,
    tta_hflip: bool,
    col2idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    val embedding 抽出（必要ならhflip TTA）
    - EMAがあればEMAでembedding
    - embeddingはL2 normalizeして返す
    """
    module.eval()
    embs = []
    ys = []

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["num"].to(device, non_blocking=True)
        y = batch["label"].detach().cpu().numpy().astype(int)

        # n_players（datasetが返さない場合に備える）
        n_players = batch.get("n_players", None)
        if n_players is None:
            if "n_players" in col2idx:
                n_players = tab[:, col2idx["n_players"]].round().long()
            else:
                n_players = torch.full((img.size(0),), 10, device=device, dtype=torch.long)
        else:
            n_players = n_players.to(device, non_blocking=True).long()

        # ---- orig ----
        if module.use_ema and getattr(module, "model_ema", None) is not None:
            e1 = module.model_ema.module.get_embedding(img, tab)
        else:
            e1 = module.model.get_embedding(img, tab)
        e1 = F.normalize(e1, p=2, dim=1)

        if not tta_hflip:
            embs.append(e1.cpu())
            ys.append(y)
            continue

        # ---- flip ----
        img_f = torch.flip(img, dims=[3])  # (B,C,H,W)
        tab_f = flip_num_feats_batch(tab, n_players, col2idx)

        if module.use_ema and getattr(module, "model_ema", None) is not None:
            e2 = module.model_ema.module.get_embedding(img_f, tab_f)
        else:
            e2 = module.model.get_embedding(img_f, tab_f)
        e2 = F.normalize(e2, p=2, dim=1)

        e = F.normalize((e1 + e2) * 0.5, p=2, dim=1)
        embs.append(e.cpu())
        ys.append(y)

    return torch.cat(embs, dim=0).numpy(), np.concatenate(ys, axis=0)


@torch.no_grad()
def compute_prototypes(
    module: PlayerLightningModule,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    """
    train loader からクラスプロトタイプ（平均embedding）を作る
    - EMAがあればEMA
    - L2 normalize
    """
    module.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["num"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True).long()

        if module.use_ema and getattr(module, "model_ema", None) is not None:
            emb = module.model_ema.module.get_embedding(img, tab)
        else:
            emb = module.model.get_embedding(img, tab)
        emb = F.normalize(emb, p=2, dim=1)

        if sums is None:
            sums = torch.zeros(num_classes, emb.size(1), dtype=emb.dtype, device=device)

        # sums[c] += emb[i] for y[i]==c
        sums.index_add_(0, y, emb)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))

    prot = sums / counts.clamp(min=1.0).unsqueeze(1)
    prot = F.normalize(prot, p=2, dim=1)
    return prot.detach().cpu().numpy()


# ===================================
# main
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg["100_train_arcface"], resolve=True)
    config = dict_to_namespace(config_dict)

    pl.seed_everything(config.seed, workers=True)
    if config.debug:
        config.exp = "100_train_arcface_debug"

    savedir = Path(config.output_dir) / config.exp
    (savedir / "oof").mkdir(parents=True, exist_ok=True)
    (savedir / "yaml").mkdir(parents=True, exist_ok=True)
    (savedir / "model").mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, savedir / "yaml" / "config.yaml")

    # ---- resolve pp paths ----
    train_pp_csv, test_pp_csv = _resolve_pp_paths(config)
    train_pp_csv = Path(train_pp_csv)
    test_pp_csv = Path(test_pp_csv)
    print("[pp] train_pp_csv:", train_pp_csv)
    print("[pp] test_pp_csv :", test_pp_csv)

    # ---- drop junk (recommended) ----
    drop_junk = bool(getattr(config, "drop_junk", True))
    junk_col = str(getattr(config, "junk_col", "is_junk"))
    train_pp_csv_for_train = _maybe_drop_junk_and_save(
        train_pp_csv=train_pp_csv,
        savedir=savedir,
        junk_col=junk_col,
        drop_junk=drop_junk,
    )

    # ---- wandb ----
    wandb_logger = None
    if config.use_wandb:
        set_wandb(config)
        wandb_logger = WandbLogger(
            project=config.wandb_project,
            name=config.exp,
            save_dir=str(savedir),
            log_model=True,
        )
        wandb_logger.log_hyperparams(config_dict)

    # ---- datamodule ----
    num_features = list(config.num_features) if hasattr(config, "num_features") else []
    dm = PlayerDataModule(
        train_pp_csv=Path(train_pp_csv_for_train),
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
        dirpath=str(savedir / "model"),
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

    # ==========================================
    # OOF & prototypes (best ckpt, no augmentation)
    # ==========================================
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # load best ckpt for evaluation consistency
    best_path = ckpt.best_model_path
    if best_path:
        module_best = PlayerLightningModule.load_from_checkpoint(
            checkpoint_path=str(best_path),
            model_name=config.model_name,
            num_classes=config.num_classes,
            num_tabular_features=len(num_features) if getattr(config, "use_tabular", True) else 0,
            embedding_dim=config.embedding_dim,
            pretrained=False,
            arcface_s=config.arcface_s,
            arcface_m=config.arcface_m,
            lr=config.lr,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay,
        ).to(device)
        module_best.eval()
        # EMA初期化（300と同様）
        module_best.setup("fit")
    else:
        module_best = module.to(device)
        module_best.eval()
        module_best.setup("fit")

    # build no-aug datasets from dm's split dfs
    val_tf = get_val_transform(config.img_size)

    # dm.setup is already called by trainer.fit, so dm.train_df / dm.val_df should exist
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

    train_loader = DataLoader(
        train_ds_noaug,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds_noaug,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    col2idx = {c: i for i, c in enumerate(num_features)}
    oof_tta_hflip = bool(getattr(config, "oof_tta_hflip", False))

    # embeddings (val)
    emb, y = extract_embeddings(module_best, val_loader, device, tta_hflip=oof_tta_hflip, col2idx=col2idx)

    oof_dir = Path(savedir) / "oof"
    np.save(oof_dir / "val_emb.npy", emb)
    np.save(oof_dir / "val_y.npy", y)

    # prototypes (train)
    prototypes = compute_prototypes(module_best, train_loader, device, num_classes=config.num_classes)
    np.save(oof_dir / "prototypes.npy", prototypes)

    # cosine similarity (emb/prototypes are already L2-normalized)
    scores = emb @ prototypes.T  # (N, C)

    top2_idx = np.argpartition(-scores, kth=1, axis=1)[:, :2]
    top2_val = np.take_along_axis(scores, top2_idx, axis=1)
    order = np.argsort(-top2_val, axis=1)

    top1 = top2_val[np.arange(len(scores)), order[:, 0]]
    top2 = top2_val[np.arange(len(scores)), order[:, 1]]
    pred = top2_idx[np.arange(len(scores)), order[:, 0]]
    margin = top1 - top2

    oof_df = pd.DataFrame({
        "y": y.astype(int),
        "pred": pred.astype(int),
        "max_sim": top1.astype(float),
        "second_sim": top2.astype(float),
        "margin": margin.astype(float),
    })
    oof_df.to_csv(oof_dir / "oof_df.csv", index=False)
    print("saved oof:", oof_dir / "oof_df.csv")


if __name__ == "__main__":
    main()