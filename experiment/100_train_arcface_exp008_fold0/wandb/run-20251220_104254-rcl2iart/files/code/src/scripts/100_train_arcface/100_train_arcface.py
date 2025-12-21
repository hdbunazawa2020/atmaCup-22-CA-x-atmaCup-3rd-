import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    EarlyStopping,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import WandbLogger

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

torch.set_float32_matmul_precision("high")  # "high" or "medium"


# ============================
# noisy warnings/logs suppress
# ============================
def _setup_noise_control():
    # ---- python warnings (only "noisy" known ones) ----
    warnings.filterwarnings(
        "ignore",
        message=r"You are sending unauthenticated requests to the HF Hub.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"ShiftScaleRotate is a special case of Affine transform.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Precision 16-mixed is not supported by the model summary.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Found \d+ module\(s\) in eval mode at the start of training.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Checkpoint directory .* exists and is not empty.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"It is recommended to use `self\.log$begin:math:text$\'\.\*\'\, \\\.\\\.\\\.\, sync\_dist\=True$end:math:text$`.*",
    )

    # ---- logging levels ----
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("timm").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


_setup_noise_control()


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main_process() -> bool:
    # single-node DDP 想定（A100x2 ならこれでOK）
    return _local_rank() == 0


def _dist_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


date = datetime.now().strftime("%Y%m%d")
if _is_main_process():
    print(f"TODAY is {date}")


# ===================================
# utils
# ===================================
def _resolve_pp_paths(config) -> tuple[Path, Path]:
    if hasattr(config, "train_pp_csv") and str(config.train_pp_csv):
        train_pp = Path(config.train_pp_csv)
    else:
        train_pp = Path(config.pp_dir) / config.pp_exp / "train_meta_pp.csv"

    if hasattr(config, "test_pp_csv") and str(config.test_pp_csv):
        test_pp = Path(config.test_pp_csv)
    else:
        test_pp = Path(config.pp_dir) / config.pp_exp / "test_meta_pp.csv"
    return train_pp, test_pp


def _maybe_drop_junk_and_save(
    train_pp_csv: Path,
    savedir: Path,
    junk_col: str = "is_junk",
    drop_junk: bool = True
) -> Path:
    if not drop_junk:
        return train_pp_csv

    df = pd.read_csv(train_pp_csv)
    if junk_col not in df.columns:
        if _is_main_process():
            print(f"[drop_junk] '{junk_col}' not in columns. skip.")
        return train_pp_csv

    before = len(df)
    m_keep = (df[junk_col].fillna(0).astype(int) == 0)
    df2 = df.loc[m_keep].copy()
    after = len(df2)

    out_csv = Path(savedir) / "train_meta_pp_clean.csv"
    # 安全のため temp -> rename（NFSでも比較的安全）
    tmp_csv = Path(savedir) / "train_meta_pp_clean.csv.tmp"
    df2.to_csv(tmp_csv, index=False)
    tmp_csv.replace(out_csv)

    if _is_main_process():
        print(f"[drop_junk] {before} -> {after} (dropped {before-after}) saved: {out_csv}")
        if "crop_variant" in df.columns:
            print("[drop_junk] crop_variant (before):\n", df["crop_variant"].value_counts())
        if "crop_variant" in df2.columns:
            print("[drop_junk] crop_variant (after):\n", df2["crop_variant"].value_counts())
    return out_csv


def flip_num_feats_batch(x: torch.Tensor, n_players: torch.Tensor, col2idx: dict[str, int]) -> torch.Tensor:
    x = x.clone()
    if "fx" in col2idx:
        x[:, col2idx["fx"]] = -x[:, col2idx["fx"]]
    if "rank_x" in col2idx:
        rx = x[:, col2idx["rank_x"]]
        x[:, col2idx["rank_x"]] = (n_players.float() - 1.0) - rx
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
    was_training = module.training
    module.eval()

    embs = []
    ys = []

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["num"].to(device, non_blocking=True)
        y = batch["label"].detach().cpu().numpy().astype(int)

        n_players = batch.get("n_players", None)
        if n_players is None:
            if "n_players" in col2idx:
                n_players = tab[:, col2idx["n_players"]].round().long()
            else:
                n_players = torch.full((img.size(0),), 10, device=device, dtype=torch.long)
        else:
            n_players = n_players.to(device, non_blocking=True).long()

        if module.use_ema and getattr(module, "model_ema", None) is not None:
            e1 = module.model_ema.module.get_embedding(img, tab)
        else:
            e1 = module.model.get_embedding(img, tab)
        e1 = F.normalize(e1, p=2, dim=1)

        if not tta_hflip:
            embs.append(e1.cpu())
            ys.append(y)
            continue

        img_f = torch.flip(img, dims=[3])
        tab_f = flip_num_feats_batch(tab, n_players, col2idx)

        if module.use_ema and getattr(module, "model_ema", None) is not None:
            e2 = module.model_ema.module.get_embedding(img_f, tab_f)
        else:
            e2 = module.model.get_embedding(img_f, tab_f)
        e2 = F.normalize(e2, p=2, dim=1)

        e = F.normalize((e1 + e2) * 0.5, p=2, dim=1)
        embs.append(e.cpu())
        ys.append(y)

    if was_training:
        module.train()

    return torch.cat(embs, dim=0).numpy(), np.concatenate(ys, axis=0)


@torch.no_grad()
def compute_prototypes(
    module: PlayerLightningModule,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    was_training = module.training
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

        sums.index_add_(0, y, emb)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))

    prot = sums / counts.clamp(min=1.0).unsqueeze(1)
    prot = F.normalize(prot, p=2, dim=1)

    if was_training:
        module.train()

    return prot.detach().cpu().numpy()


# ===================================
# threshold-aware CV utilities
# ===================================
def _top2_from_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, C = scores.shape
    top2_idx = np.argpartition(-scores, kth=1, axis=1)[:, :2]
    top2_val = np.take_along_axis(scores, top2_idx, axis=1)
    order = np.argsort(-top2_val, axis=1)

    max_sim = top2_val[np.arange(N), order[:, 0]]
    second_sim = top2_val[np.arange(N), order[:, 1]]
    pred_class = top2_idx[np.arange(N), order[:, 0]]
    margin = max_sim - second_sim
    return pred_class.astype(int), max_sim.astype(float), second_sim.astype(float), margin.astype(float)


def _cm_with_unknown(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    K = num_classes + 1
    y = y_true.astype(int)
    p = y_pred.astype(int)
    y_idx = y + 1
    p_idx = p + 1

    flat = y_idx * K + p_idx
    cm = np.bincount(flat, minlength=K * K).reshape(K, K)
    return cm


def _macro_f1_holdout_from_cm(
    cm: np.ndarray,
    y_present_base: np.ndarray,
    pred_present: np.ndarray,
    holdout_id: int,
) -> float:
    K = cm.shape[0]
    C = K - 1
    hid = int(holdout_id)
    if not (0 <= hid < C):
        return float("nan")

    idxh = hid + 1

    rsum = cm.sum(axis=1).astype(np.float64)
    csum = cm.sum(axis=0).astype(np.float64)
    diag = np.diag(cm).astype(np.float64)

    tp = diag.copy()
    rsum_adj = rsum.copy()

    rsum_adj[0] = rsum[0] + rsum[idxh]
    tp[0] = diag[0] + cm[idxh, 0]

    rsum_adj[idxh] = 0.0
    tp[idxh] = 0.0

    y_present = y_present_base.copy()
    if y_present_base[idxh]:
        y_present[0] = True
    y_present[idxh] = False
    union = y_present | pred_present

    fp = csum - tp
    fn = rsum_adj - tp
    denom = 2.0 * tp + fp + fn

    f1 = np.zeros_like(tp, dtype=np.float64)
    np.divide(2.0 * tp, denom, out=f1, where=(denom > 0))

    if not union.any():
        return 0.0
    return float(f1[union].mean())


def optimize_threshold_holdout_mean(
    y_true: np.ndarray,
    scores: np.ndarray,
    holdout_ids: list[int],
    thr_values: np.ndarray,
    mthr_values: np.ndarray,
    combine_mode: str = "or",
) -> dict:
    y_true = y_true.astype(int)
    C = scores.shape[1]

    pred_class, max_sim, second_sim, margin = _top2_from_scores(scores)

    y_counts = np.bincount(y_true, minlength=C)
    y_present_base = np.zeros(C + 1, dtype=bool)  # idx0=-1
    y_present_base[1:] = (y_counts > 0)

    holdouts = [int(h) for h in holdout_ids if 0 <= int(h) < C and y_counts[int(h)] > 0]
    if len(holdouts) == 0:
        return {
            "best_score": -1e18,
            "best_thr_sim": None,
            "best_thr_margin": None,
            "unknown_rate": float("nan"),
            "holdout_ids_used": [],
        }

    best_score = -1e18
    best_thr = None
    best_mthr = None
    best_unknown_rate = None

    for thr in thr_values:
        cond_sim = (max_sim < float(thr))
        for mthr in mthr_values:
            if combine_mode == "and":
                unk = cond_sim & (margin < float(mthr))
            else:
                unk = cond_sim | (margin < float(mthr))

            pred = np.where(unk, -1, pred_class).astype(int)
            cm = _cm_with_unknown(y_true, pred, num_classes=C)
            pred_present = (np.bincount(pred + 1, minlength=C + 1) > 0)

            scores_h = []
            for hid in holdouts:
                s = _macro_f1_holdout_from_cm(cm, y_present_base, pred_present, hid)
                scores_h.append(s)

            mean_s = float(np.mean(scores_h))
            if mean_s > best_score:
                best_score = mean_s
                best_thr = float(thr)
                best_mthr = float(mthr)
                best_unknown_rate = float((pred == -1).mean())

    return {
        "best_score": float(best_score),
        "best_thr_sim": best_thr,
        "best_thr_margin": best_mthr,
        "unknown_rate": float(best_unknown_rate) if best_unknown_rate is not None else float("nan"),
        "holdout_ids_used": holdouts,
    }


def _try_get_weight_prototypes(pl_module: PlayerLightningModule, num_classes: int) -> Optional[np.ndarray]:
    base = pl_module.model_ema.module if (pl_module.use_ema and getattr(pl_module, "model_ema", None) is not None) else pl_module.model

    candidates = [
        ("arcface", "weight"),
        ("metric_fc", "weight"),
        ("classifier", "weight"),
        ("fc", "weight"),
        ("head", "weight"),
    ]
    for layer_name, w_name in candidates:
        layer = getattr(base, layer_name, None)
        if layer is None:
            continue
        w = getattr(layer, w_name, None)
        if w is None:
            continue
        wt = w.detach()
        if wt.ndim == 2 and wt.shape[0] == num_classes:
            wt = F.normalize(wt, p=2, dim=1)
            return wt.cpu().numpy()
    return None


class ThresholdAwareCVCallback(pl.Callback):
    """
    ✅ 重要：
    - ModelCheckpoint(monitor='val_thr_cv') が参照するのは trainer.callback_metrics
    - callback からの pl_module.log() だけだと載らないケースがあるので
      trainer.callback_metrics に明示的に注入する
    - DDP なので rank0 で計算 → broadcast → 全rankで同じキーをセット
    """
    def __init__(
        self,
        savedir: Path,
        num_classes: int,
        num_features: list[str],
        img_size: int,
        batch_size: int,
        num_workers: int,
        tta_hflip: bool,
        holdout_ids: list[int],
        thr_min: float,
        thr_max: float,
        thr_step: float,
        mthr_min: float,
        mthr_max: float,
        mthr_step: float,
        combine_mode: str = "or",
        every_n_epochs: int = 1,
        prototype_source: str = "weights",  # "weights" or "train"
        save_json: bool = True,
    ):
        super().__init__()
        self.savedir = Path(savedir)
        self.num_classes = int(num_classes)
        self.num_features = list(num_features)
        self.img_size = int(img_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.tta_hflip = bool(tta_hflip)
        self.holdout_ids = list(holdout_ids)
        self.combine_mode = str(combine_mode)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.prototype_source = str(prototype_source)
        self.save_json = bool(save_json)

        self.thr_values = np.arange(float(thr_min), float(thr_max) + 1e-9, float(thr_step))
        self.mthr_values = np.arange(float(mthr_min), float(mthr_max) + 1e-9, float(mthr_step))

        self.col2idx = {c: i for i, c in enumerate(self.num_features)}

        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None

        self.outdir = self.savedir / "threshold_cv"
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.best_so_far = -1e18
        self.last_payload: Optional[dict] = None  # every_n_epochs>1 対策（キー欠落防止）

    def _build_loaders_if_needed(self, dm: PlayerDataModule):
        if self._val_loader is None:
            tf = get_val_transform(self.img_size)
            val_ds = PlayerDataset(
                df=dm.val_df,
                transform=tf,
                num_features=self.num_features,
                is_train=False,
                p_hflip=0.0,
            )
            self._val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=(self.num_workers > 0),
            )

        if self.prototype_source == "train" and self._train_loader is None:
            tf = get_val_transform(self.img_size)
            train_ds = PlayerDataset(
                df=dm.train_df,
                transform=tf,
                num_features=self.num_features,
                is_train=False,
                p_hflip=0.0,
            )
            self._train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=(self.num_workers > 0),
            )

    @staticmethod
    def _inject_metric(trainer: pl.Trainer, name: str, value: float, prog_bar: bool = False):
        # callback_metrics / logged_metrics / progress_bar_metrics へ明示的に入れる
        v = torch.tensor(float(value))
        trainer.callback_metrics[name] = v
        trainer.logged_metrics[name] = v
        if prog_bar:
            trainer.progress_bar_metrics[name] = v

    def on_validation_end(self, trainer: pl.Trainer, pl_module: PlayerLightningModule) -> None:
        # sanity check は重いのでスキップ（Trainer側でも num_sanity_val_steps=0 にする）
        if getattr(trainer, "sanity_checking", False):
            return

        epoch = int(trainer.current_epoch)
        need_compute = ((epoch + 1) % self.every_n_epochs) == 0

        # -------- rank0 compute --------
        payload = None
        if need_compute and trainer.is_global_zero:
            dm = trainer.datamodule
            if dm is None or (not hasattr(dm, "train_df")) or (not hasattr(dm, "val_df")):
                payload = None
            else:
                self._build_loaders_if_needed(dm)

                t0 = time.time()
                device = pl_module.device

                # 1) val embeddings
                val_emb, val_y = extract_embeddings(
                    pl_module,
                    self._val_loader,
                    device=device,
                    tta_hflip=self.tta_hflip,
                    col2idx=self.col2idx,
                )

                # 2) prototypes
                prototypes = None
                if self.prototype_source == "weights":
                    prototypes = _try_get_weight_prototypes(pl_module, num_classes=self.num_classes)
                    if prototypes is None:
                        if _is_main_process():
                            print("[thr_cv] prototype_source=weights but weight not found -> fallback to train prototypes")
                        self.prototype_source = "train"
                        self._build_loaders_if_needed(dm)

                if prototypes is None:
                    prototypes = compute_prototypes(
                        pl_module,
                        self._train_loader,
                        device=device,
                        num_classes=self.num_classes,
                    )

                # 3) scores
                scores = val_emb @ prototypes.T

                # 4) optimize
                best = optimize_threshold_holdout_mean(
                    y_true=val_y,
                    scores=scores,
                    holdout_ids=self.holdout_ids,
                    thr_values=self.thr_values,
                    mthr_values=self.mthr_values,
                    combine_mode=self.combine_mode,
                )

                elapsed = time.time() - t0
                val_thr_cv = float(best["best_score"])
                thr_sim = best["best_thr_sim"]
                thr_m = best["best_thr_margin"]
                unk_rate = float(best["unknown_rate"])

                payload = {
                    "epoch": epoch,
                    "val_thr_cv": val_thr_cv,
                    "thr_sim": float(thr_sim) if thr_sim is not None else float("nan"),
                    "thr_margin": float(thr_m) if thr_m is not None else float("nan"),
                    "unknown_rate": float(unk_rate),
                    "elapsed_sec": float(elapsed),
                    "combine_mode": self.combine_mode,
                    "holdout_ids_used": best.get("holdout_ids_used", []),
                    "prototype_source": self.prototype_source,
                    "tta_hflip": self.tta_hflip,
                }

                self.last_payload = payload

                print(
                    f"[thr_cv][epoch={epoch}] score={val_thr_cv:.6f} "
                    f"thr_sim={payload['thr_sim']} thr_margin={payload['thr_margin']} unk_rate={unk_rate:.4f} "
                    f"prototype_source={self.prototype_source} time={elapsed:.2f}s"
                )

                # save json
                if self.save_json:
                    (self.outdir / "latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                    (self.outdir / f"epoch_{epoch:03d}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                    if val_thr_cv > self.best_so_far:
                        self.best_so_far = val_thr_cv
                        (self.outdir / "best.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        # -------- ensure key exists every epoch (for ckpt monitor) --------
        if (payload is None) and (self.last_payload is not None):
            payload = dict(self.last_payload)  # shallow copy
            payload["epoch"] = epoch  # current epoch
        if payload is None:
            payload = {
                "epoch": epoch,
                "val_thr_cv": -1e18,  # monitor 用にキー欠落させない
                "thr_sim": float("nan"),
                "thr_margin": float("nan"),
                "unknown_rate": float("nan"),
                "elapsed_sec": 0.0,
            }

        # -------- broadcast to all ranks --------
        # [val_thr_cv, thr_sim, thr_margin, unk_rate, elapsed]
        if trainer.is_global_zero:
            t = torch.tensor(
                [payload["val_thr_cv"], payload["thr_sim"], payload["thr_margin"], payload["unknown_rate"], payload["elapsed_sec"]],
                device=pl_module.device,
                dtype=torch.float32,
            )
        else:
            t = torch.zeros(5, device=pl_module.device, dtype=torch.float32)

        if _dist_ready():
            torch.distributed.broadcast(t, src=0)

        vals = t.detach().cpu().tolist()
        val_thr_cv, thr_sim, thr_margin, unk_rate, elapsed = [float(x) for x in vals]

        # ✅ ここが今回の核心：ModelCheckpoint が見る dict に必ず入れる
        self._inject_metric(trainer, "val_thr_cv", val_thr_cv, prog_bar=True)
        self._inject_metric(trainer, "val_thr_sim", thr_sim, prog_bar=False)
        self._inject_metric(trainer, "val_thr_margin", thr_margin, prog_bar=False)
        self._inject_metric(trainer, "val_thr_unknown_rate", unk_rate, prog_bar=False)
        self._inject_metric(trainer, "val_thr_cv_time_sec", elapsed, prog_bar=False)

        # wandb へも（rank0 のみ）
        if trainer.is_global_zero and trainer.logger:
            try:
                trainer.logger.log_metrics(
                    {
                        "val_thr_cv": val_thr_cv,
                        "val_thr_sim": thr_sim,
                        "val_thr_margin": thr_margin,
                        "val_thr_unknown_rate": unk_rate,
                        "val_thr_cv_time_sec": elapsed,
                    },
                    step=trainer.global_step,
                )
            except Exception:
                pass


# ===================================
# main
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    base_dict = OmegaConf.to_container(cfg["100_train_arcface"], resolve=True)
    base = dict_to_namespace(base_dict)

    pl.seed_everything(base.seed, workers=True)

    folds = list(getattr(base, "folds", [int(getattr(base, "fold", 0))]))

    exp_base = str(getattr(base, "exp", "100_train_arcface_exp"))
    for suf in ["_fold0", "_fold1", "_fold2", "_fold3", "_fold4"]:
        if exp_base.endswith(suf):
            exp_base = exp_base[: -len(suf)]

    if _is_main_process():
        print(f"[folds] will run folds={folds}, exp_base={exp_base}")

    for fold in folds:
        fold = int(fold)

        config_dict = dict(base_dict)
        config_dict["fold"] = fold
        config_dict["exp"] = f"{exp_base}_fold{fold}"
        config = dict_to_namespace(config_dict)

        if config.debug:
            config.exp = "100_train_arcface_debug"

        savedir = Path(config.output_dir) / config.exp
        (savedir / "oof").mkdir(parents=True, exist_ok=True)
        (savedir / "yaml").mkdir(parents=True, exist_ok=True)
        (savedir / "model").mkdir(parents=True, exist_ok=True)

        if _is_main_process():
            save_config_yaml(config, savedir / "yaml" / "config.yaml")

            print(f"\n==============================")
            print(f"[RUN] exp={config.exp} fold={fold}/{len(folds)}")
            print(f"==============================")

        # pp paths / drop junk
        train_pp_csv, test_pp_csv = _resolve_pp_paths(config)
        if _is_main_process():
            print("[pp] train_pp_csv:", train_pp_csv)
            print("[pp] test_pp_csv :", test_pp_csv)

        drop_junk = bool(getattr(config, "drop_junk", True))
        junk_col = str(getattr(config, "junk_col", "is_junk"))

        # ✅ 競合回避：rank0だけ作る、他rankは出来上がるまで待つ
        clean_csv = Path(savedir) / "train_meta_pp_clean.csv"
        if drop_junk:
            if _is_main_process():
                train_pp_csv_for_train = _maybe_drop_junk_and_save(
                    train_pp_csv=Path(train_pp_csv),
                    savedir=savedir,
                    junk_col=junk_col,
                    drop_junk=True,
                )
            else:
                train_pp_csv_for_train = clean_csv
                # wait for rank0 to write
                for _ in range(1200):  # 0.25s * 1200 = 300s
                    if clean_csv.exists():
                        break
                    time.sleep(0.25)
                if not clean_csv.exists():
                    raise RuntimeError(f"[drop_junk] clean csv not found: {clean_csv}")
        else:
            train_pp_csv_for_train = Path(train_pp_csv)

        # wandb (rank0 only)
        wandb_logger = None
        use_wandb = bool(getattr(config, "use_wandb", False)) and _is_main_process()
        if use_wandb:
            set_wandb(config)
            wandb_logger = WandbLogger(
                project=config.wandb_project,
                name=config.exp,
                save_dir=str(savedir),
                log_model=True,
            )
            wandb_logger.log_hyperparams(config_dict)

        # datamodule
        num_features = list(config.num_features) if hasattr(config, "num_features") else []
        dm = PlayerDataModule(
            train_pp_csv=Path(train_pp_csv_for_train),
            img_size=config.img_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            num_features=num_features,
            split_method=str(getattr(config, "split_method", "holdout")),
            val_quarter_from=getattr(config, "val_quarter_from", "Q2-016"),
            n_splits=int(getattr(config, "n_splits", 5)),
            fold=fold,
            seed=config.seed,
            p_hflip=getattr(config, "p_hflip", 0.5),
        )

        # module
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

        ckpt_monitor = str(getattr(config, "ckpt_monitor", "val_f1"))
        ckpt_mode = str(getattr(config, "ckpt_mode", "max"))

        # safeguard: monitor が val_thr_cv なのに threshold_cv_enable=False だと必ず落ちるので保険
        thr_enable = bool(getattr(config, "threshold_cv_enable", False))
        if (ckpt_monitor == "val_thr_cv") and (not thr_enable):
            if _is_main_process():
                print("[WARN] ckpt_monitor=val_thr_cv but threshold_cv_enable=false -> fallback to val_f1")
            ckpt_monitor = "val_f1"

        ckpt = ModelCheckpoint(
            dirpath=str(savedir / "model"),
            filename="best",
            monitor=ckpt_monitor,
            mode=ckpt_mode,
            save_top_k=1,
            save_last=True,
            verbose=_is_main_process(),
            enable_version_counter=False,
            save_on_train_epoch_end=False,  # validation end で保存
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

        callbacks = []

        # threshold-aware CV callback（ckpt より先に）
        if thr_enable:
            thr_cb = ThresholdAwareCVCallback(
                savedir=savedir,
                num_classes=int(config.num_classes),
                num_features=num_features,
                img_size=int(config.img_size),
                batch_size=int(getattr(config, "threshold_cv_batch_size", config.batch_size)),
                num_workers=int(getattr(config, "threshold_cv_num_workers", config.num_workers)),
                tta_hflip=bool(getattr(config, "threshold_cv_tta_hflip", getattr(config, "oof_tta_hflip", False))),
                holdout_ids=list(getattr(config, "threshold_cv_holdout_ids", list(range(int(config.num_classes))))),
                thr_min=float(getattr(config, "threshold_cv_thr_min", 0.5)),
                thr_max=float(getattr(config, "threshold_cv_thr_max", 1.0)),
                thr_step=float(getattr(config, "threshold_cv_thr_step", 0.01)),
                mthr_min=float(getattr(config, "threshold_cv_mthr_min", 0.0)),
                mthr_max=float(getattr(config, "threshold_cv_mthr_max", 0.10)),
                mthr_step=float(getattr(config, "threshold_cv_mthr_step", 0.005)),
                combine_mode=str(getattr(config, "threshold_cv_combine_mode", "or")),
                every_n_epochs=int(getattr(config, "threshold_cv_every_n_epochs", 1)),
                prototype_source=str(getattr(config, "threshold_cv_prototype_source", "weights")),
                save_json=bool(getattr(config, "threshold_cv_save_json", True)),
            )
            callbacks.append(thr_cb)

        callbacks += [ckpt, pbar]

        if use_wandb:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        if bool(getattr(config, "use_early_stopping", False)):
            patience = int(getattr(config, "early_stopping_patience", 5))
            callbacks.append(EarlyStopping(monitor=ckpt_monitor, mode=ckpt_mode, patience=patience))

        trainer = pl.Trainer(
            max_epochs=config.epochs,
            accelerator="auto",
            devices="auto",
            strategy="ddp_find_unused_parameters_true",
            callbacks=callbacks,
            logger=wandb_logger if use_wandb else False,
            precision="16-mixed",
            deterministic=True,
            default_root_dir=str(savedir),
            num_sanity_val_steps=0,         # ✅ sanity check を消す（無駄+ログ増える）
            enable_model_summary=False,     # ✅ model_summary警告を消す
        )

        trainer.fit(module, dm)

        # ---- post fit: OOF/prototypes ----
        # ✅ DDPだとここも各rankで走るので、rank0だけ実行
        if not trainer.is_global_zero:
            continue

        print("[DONE] best ckpt:", ckpt.best_model_path)

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
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
        else:
            module_best = module.to(device)
            module_best.eval()

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

        train_loader = DataLoader(
            train_ds_noaug,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(config.num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds_noaug,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(config.num_workers > 0),
        )

        col2idx = {c: i for i, c in enumerate(num_features)}
        oof_tta_hflip = bool(getattr(config, "oof_tta_hflip", False))

        emb, y = extract_embeddings(
            module_best,
            val_loader,
            device=device,
            tta_hflip=oof_tta_hflip,
            col2idx=col2idx,
        )

        oof_dir = Path(savedir) / "oof"
        np.save(oof_dir / f"val_emb_fold{fold}.npy", emb)
        np.save(oof_dir / f"val_y_fold{fold}.npy", y)

        prototypes = compute_prototypes(module_best, train_loader, device, num_classes=config.num_classes)
        np.save(oof_dir / f"prototypes_fold{fold}.npy", prototypes)

        scores = emb @ prototypes.T
        pred, top1, top2, margin = _top2_from_scores(scores)

        oof_df = pd.DataFrame({
            "y": y.astype(int),
            "pred": pred.astype(int),
            "max_sim": top1.astype(float),
            "second_sim": top2.astype(float),
            "margin": margin.astype(float),
        })
        oof_df.to_csv(oof_dir / f"oof_df_fold{fold}.csv", index=False)
        print("saved oof:", oof_dir / f"oof_df_fold{fold}.csv")


if __name__ == "__main__":
    main()