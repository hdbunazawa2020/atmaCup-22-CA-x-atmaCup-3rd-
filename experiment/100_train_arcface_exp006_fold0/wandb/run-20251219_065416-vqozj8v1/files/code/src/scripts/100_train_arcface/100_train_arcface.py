import os
import json
import time
from pathlib import Path
from datetime import datetime

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
# utils (existing)
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
        print(f"[drop_junk] '{junk_col}' not in columns. skip.")
        return train_pp_csv

    before = len(df)
    m_keep = (df[junk_col].fillna(0).astype(int) == 0)
    df2 = df.loc[m_keep].copy()
    after = len(df2)

    out_csv = Path(savedir) / "train_meta_pp_clean.csv"
    df2.to_csv(out_csv, index=False)

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

    return torch.cat(embs, dim=0).numpy(), np.concatenate(ys, axis=0)


@torch.no_grad()
def compute_prototypes(
    module: PlayerLightningModule,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
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
    return prot.detach().cpu().numpy()


# ===================================
# threshold-aware CV (NEW)
# ===================================
def _top2_from_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    scores: (N,C) cosine similarity
    return:
      pred_class (N,)
      max_sim (N,)
      second_sim (N,)
      margin (N,)
    """
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
    """
    labels mapping:
      idx 0 -> -1
      idx 1..C -> 0..C-1
    """
    K = num_classes + 1
    y = y_true.astype(int)
    p = y_pred.astype(int)
    # map -1 -> 0, class -> +1
    y_idx = y + 1
    p_idx = p + 1

    if (y_idx < 0).any() or (y_idx >= K).any():
        raise ValueError("y_true contains out-of-range labels (expected -1..C-1)")
    if (p_idx < 0).any() or (p_idx >= K).any():
        raise ValueError("y_pred contains out-of-range labels (expected -1..C-1)")

    flat = y_idx * K + p_idx
    cm = np.bincount(flat, minlength=K * K).reshape(K, K)
    return cm


def _macro_f1_holdout_from_cm(
    cm: np.ndarray,
    y_present_base: np.ndarray,
    pred_present: np.ndarray,
    holdout_id: int,
) -> float:
    """
    y_true の holdout_id を -1 とみなした macro-F1 を、cm から高速に計算する。

    cm は「元の y_true(=0..C-1) vs y_pred(-1..C-1)」の confusion matrix。
    holdout_id の行を -1 行にマージして評価する。
    """
    K = cm.shape[0]
    C = K - 1
    hid = int(holdout_id)
    if not (0 <= hid < C):
        return float("nan")

    idxh = hid + 1  # row/col index for class hid

    rsum = cm.sum(axis=1).astype(np.float64)
    csum = cm.sum(axis=0).astype(np.float64)
    diag = np.diag(cm).astype(np.float64)

    # adjust for y: hid -> -1
    tp = diag.copy()
    rsum_adj = rsum.copy()

    # move row idxh to row 0
    rsum_adj[0] = rsum[0] + rsum[idxh]
    tp[0] = diag[0] + cm[idxh, 0]

    # hid row becomes empty
    rsum_adj[idxh] = 0.0
    tp[idxh] = 0.0

    # union labels = (y_present_after | pred_present)
    y_present = y_present_base.copy()
    if y_present_base[idxh]:
        y_present[0] = True
    y_present[idxh] = False
    union = y_present | pred_present

    fp = csum - tp
    fn = rsum_adj - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.where(denom > 0, (2.0 * tp) / denom, 0.0)

    # union のみ macro
    return float(f1[union].mean())


def optimize_threshold_holdout_mean(
    y_true: np.ndarray,
    scores: np.ndarray,
    holdout_ids: list[int],
    thr_values: np.ndarray,
    mthr_values: np.ndarray,
    combine_mode: str = "or",
) -> dict:
    """
    global threshold を探索:
      score(thr, mthr) = mean_{hid in holdout_ids} macroF1(y(hid->-1), pred(thr,mthr))

    return: best thresholds + best score
    """
    y_true = y_true.astype(int)
    C = scores.shape[1]

    pred_class, max_sim, second_sim, margin = _top2_from_scores(scores)

    # present mask in original y (no -1)
    y_counts = np.bincount(y_true, minlength=C)
    y_present_base = np.zeros(C + 1, dtype=bool)  # idx0=-1
    y_present_base[1:] = (y_counts > 0)

    holdouts = [int(h) for h in holdout_ids if 0 <= int(h) < C and y_counts[int(h)] > 0]
    if len(holdouts) == 0:
        return {
            "best_score": float("nan"),
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
                # default: or
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


def _try_get_weight_prototypes(pl_module: PlayerLightningModule, num_classes: int) -> np.ndarray | None:
    """
    ArcFace の class weight を prototype として取り出す（高速モード）。
    モデル実装に依存するので、いくつかの候補を試して見つかれば返す。
    """
    # use EMA weights if available
    base = None
    if pl_module.use_ema and getattr(pl_module, "model_ema", None) is not None:
        base = pl_module.model_ema.module
    else:
        base = pl_module.model

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
        if isinstance(w, torch.Tensor) or hasattr(w, "detach"):
            wt = w.detach()
            if wt.ndim == 2 and wt.shape[0] == num_classes:
                wt = F.normalize(wt, p=2, dim=1)
                return wt.cpu().numpy()

    return None


class ThresholdAwareCVCallback(pl.Callback):
    """
    validation epoch end で:
      - val embeddings (optionally hflip TTA)
      - prototypes (weights or train)
      - threshold grid search (pseudo-unknown mean macroF1)
    を行い、val_thr_cv を log する
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

        self._train_loader = None
        self._val_loader = None

        self.outdir = self.savedir / "threshold_cv"
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.best_so_far = -1e18

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
            )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: PlayerLightningModule) -> None:
        if not trainer.is_global_zero:
            return

        epoch = int(trainer.current_epoch)
        if ((epoch + 1) % self.every_n_epochs) != 0:
            return

        dm = trainer.datamodule
        if dm is None:
            return
        if not hasattr(dm, "train_df") or not hasattr(dm, "val_df"):
            return

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
                print("[thr_cv] prototype_source=weights but weight not found -> fallback to train prototypes")
                if self._train_loader is None:
                    self.prototype_source = "train"
                    self._build_loaders_if_needed(dm)

        if prototypes is None:
            prototypes = compute_prototypes(
                pl_module,
                self._train_loader,
                device=device,
                num_classes=self.num_classes,
            )

        # 3) similarity scores
        scores = val_emb @ prototypes.T  # (N,C)

        # 4) optimize thresholds (global)
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

        # log (checkpoint/ESで使えるよう pl_module.log を使用)
        pl_module.log("val_thr_cv", val_thr_cv, prog_bar=True, on_epoch=True, logger=True, sync_dist=False)
        pl_module.log("val_thr_sim", float(thr_sim) if thr_sim is not None else np.nan, prog_bar=False, on_epoch=True, logger=True, sync_dist=False)
        pl_module.log("val_thr_margin", float(thr_m) if thr_m is not None else np.nan, prog_bar=False, on_epoch=True, logger=True, sync_dist=False)
        pl_module.log("val_thr_unknown_rate", unk_rate, prog_bar=False, on_epoch=True, logger=True, sync_dist=False)
        pl_module.log("val_thr_cv_time_sec", float(elapsed), prog_bar=False, on_epoch=True, logger=True, sync_dist=False)

        print(
            f"[thr_cv][epoch={epoch}] score={val_thr_cv:.6f} "
            f"thr_sim={thr_sim} thr_margin={thr_m} unk_rate={unk_rate:.4f} "
            f"prototype_source={self.prototype_source} time={elapsed:.2f}s"
        )

        # save json (optional)
        if self.save_json:
            payload = {
                "epoch": epoch,
                "val_thr_cv": val_thr_cv,
                "thr_sim": thr_sim,
                "thr_margin": thr_m,
                "unknown_rate": unk_rate,
                "combine_mode": self.combine_mode,
                "holdout_ids_used": best.get("holdout_ids_used", []),
                "prototype_source": self.prototype_source,
                "tta_hflip": self.tta_hflip,
                "elapsed_sec": float(elapsed),
            }
            (self.outdir / "latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

            ep_path = self.outdir / f"epoch_{epoch:03d}.json"
            ep_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

            if val_thr_cv > self.best_so_far:
                self.best_so_far = val_thr_cv
                (self.outdir / "best.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# ===================================
# main
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    base_dict = OmegaConf.to_container(cfg["100_train_arcface"], resolve=True)
    base = dict_to_namespace(base_dict)

    pl.seed_everything(base.seed, workers=True)

    # folds を優先。無ければ単発 fold=0 を実行
    folds = list(getattr(base, "folds", [int(getattr(base, "fold", 0))]))

    # exp は “ベース名” 扱いにして fold を付ける
    exp_base = str(getattr(base, "exp", "100_train_arcface_exp"))
    # exp がすでに _foldX を含む場合は事故るので保険
    #（fold0が入ってても剥がしてベースに戻す）
    for suf in ["_fold0", "_fold1", "_fold2", "_fold3", "_fold4"]:
        if exp_base.endswith(suf):
            exp_base = exp_base[: -len(suf)]

    print(f"[folds] will run folds={folds}, exp_base={exp_base}")

    for fold in folds:
        fold = int(fold)

        # -----------------------------
        # fold ごとの config を作る
        # -----------------------------
        config_dict = dict(base_dict)  # shallow copy
        config_dict["fold"] = fold
        config_dict["exp"] = f"{exp_base}_fold{fold}"
        config = dict_to_namespace(config_dict)

        if config.debug:
            config.exp = "100_train_arcface_debug"

        # -----------------------------
        # 保存先（foldごと）
        # -----------------------------
        savedir = Path(config.output_dir) / config.exp
        (savedir / "oof").mkdir(parents=True, exist_ok=True)
        (savedir / "yaml").mkdir(parents=True, exist_ok=True)
        (savedir / "model").mkdir(parents=True, exist_ok=True)
        save_config_yaml(config, savedir / "yaml" / "config.yaml")

        print(f"\n==============================")
        print(f"[RUN] exp={config.exp} fold={fold}/{len(folds)}")
        print(f"==============================")

        # -----------------------------
        # pp paths / drop junk
        # -----------------------------
        train_pp_csv, test_pp_csv = _resolve_pp_paths(config)
        train_pp_csv = Path(train_pp_csv)
        test_pp_csv = Path(test_pp_csv)
        print("[pp] train_pp_csv:", train_pp_csv)
        print("[pp] test_pp_csv :", test_pp_csv)

        drop_junk = bool(getattr(config, "drop_junk", True))
        junk_col = str(getattr(config, "junk_col", "is_junk"))
        train_pp_csv_for_train = _maybe_drop_junk_and_save(
            train_pp_csv=train_pp_csv,
            savedir=savedir,
            junk_col=junk_col,
            drop_junk=drop_junk,
        )

        # -----------------------------
        # wandb（foldごとに別run）
        # -----------------------------
        wandb_logger = None
        if config.use_wandb:
            set_wandb(config)
            wandb_logger = WandbLogger(
                project=config.wandb_project,
                name=config.exp,   # fold入り
                save_dir=str(savedir),
                log_model=True,
            )
            # base_dict ではなく fold反映後をログ
            wandb_logger.log_hyperparams(config_dict)

        # -----------------------------
        # datamodule（foldを渡す）
        # -----------------------------
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

        # -----------------------------
        # module
        # -----------------------------
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

        # -----------------------------
        # callbacks（foldごとに作り直す）
        # -----------------------------
        ckpt_monitor = str(getattr(config, "ckpt_monitor", "val_f1"))
        ckpt_mode = str(getattr(config, "ckpt_mode", "max"))

        ckpt = ModelCheckpoint(
            dirpath=str(savedir / "model"),
            filename="best",
            monitor=ckpt_monitor,
            mode=ckpt_mode,
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

        callbacks = []

        # threshold-aware CV callback（foldごと新規）
        if bool(getattr(config, "threshold_cv_enable", False)):
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
            # monitor に使うなら ckpt より先に入れる
            callbacks.append(thr_cb)

        callbacks += [ckpt, pbar]

        if config.use_wandb:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        if bool(getattr(config, "use_early_stopping", False)):
            patience = int(getattr(config, "early_stopping_patience", 5))
            callbacks.append(EarlyStopping(monitor=ckpt_monitor, mode=ckpt_mode, patience=patience))

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

        # -----------------------------
        # fit
        # -----------------------------
        trainer.fit(module, dm)
        print("[DONE] best ckpt:", ckpt.best_model_path)

        # ==========================================
        # OOF & prototypes (best ckpt, no augmentation)
        # ==========================================
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
            module_best.setup("fit")
        else:
            module_best = module.to(device)
            module_best.eval()
            module_best.setup("fit")

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

        emb, y = extract_embeddings(module_best, val_loader, device, tta_hflip=oof_tta_hflip, col2idx=col2idx)

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