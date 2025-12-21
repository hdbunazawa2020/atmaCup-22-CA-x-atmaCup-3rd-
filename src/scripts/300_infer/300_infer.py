import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append("..")

from utils.data import save_config_yaml, dict_to_namespace
from atma_datasets.player_dataset import PlayerDataset
from atma_datasets.transforms import get_val_transform
from models.lightning_module import PlayerLightningModule

date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")


# ===================================
# utils: config / dataframe
# ===================================
def load_train_yaml(train_output_dir: str, train_exp: str) -> dict:
    p = Path(train_output_dir) / train_exp / "yaml" / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"train config yaml not found: {p}")
    return OmegaConf.to_container(OmegaConf.load(p), resolve=True)


def prepare_df(df: pd.DataFrame, num_features: list[str]) -> pd.DataFrame:
    df = df.copy()

    df["angle_id"] = (df["angle"] == "top").astype(np.float32)

    if "n_players" not in df.columns:
        df["n_players"] = (
            df.groupby(["quarter", "session", "frame", "angle"], sort=False)["angle"]
              .transform("size")
              .astype(np.int32)
        )
    df["n_players"] = df["n_players"].fillna(10).astype(int)

    for c in num_features:
        if c not in df.columns:
            raise KeyError(f"num_feature not found: {c}")
        df[c] = df[c].fillna(0.0)

    return df


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


# ===================================
# utils: prototype / embedding
# ===================================
@torch.no_grad()
def compute_prototypes(module: PlayerLightningModule, loader: DataLoader, device: torch.device, num_classes: int) -> torch.Tensor:
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
    return prot.detach().cpu()


@torch.no_grad()
def extract_embeddings(
    module: PlayerLightningModule,
    loader: DataLoader,
    device: torch.device,
    tta_hflip: bool,
    col2idx: dict[str, int],
) -> torch.Tensor:
    module.eval()
    embs = []

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["num"].to(device, non_blocking=True)

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

    return torch.cat(embs, dim=0)


# ===================================
# utils: threshold loading / building
# ===================================
def _parse_keyval_file(path: Path) -> dict[str, Any]:
    kv: dict[str, Any] = {}
    if not path.exists():
        return kv
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ("=" not in s):
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            kv[k] = float(v)
        except Exception:
            kv[k] = v
    return kv


def load_thresholds(exp_dir: Path, config) -> tuple[str, dict, dict]:
    threshold_source = str(getattr(config, "threshold_source", "auto"))

    thr_key = str(getattr(config, "thr_key", "thr_median_nonzero"))
    mthr_key = str(getattr(config, "mthr_key", "mthr_median_nonzero"))

    # (A) classwise json
    json_path = exp_dir / "threshold" / "classwise_thresholds.json"
    if threshold_source in ("auto", "classwise", "classwise_json") and json_path.exists():
        payload = json.loads(json_path.read_text())

        combine_mode = str(payload.get("combine_mode", getattr(config, "combine_mode", "or")))
        g = payload.get("global", {})
        global_thr = {
            "thr_sim": float(g.get(thr_key, g.get("thr_median_nonzero", g.get("thr_median", getattr(config, "unknown_threshold", 0.0))))),
            "thr_margin": float(g.get(mthr_key, g.get("mthr_median_nonzero", g.get("mthr_median", getattr(config, "margin_threshold", 0.0))))),
        }

        classwise = {}
        for k, v in payload.get("classwise", {}).items():
            try:
                cid = int(k)
            except Exception:
                continue
            classwise[cid] = {
                "thr_sim": float(v.get("thr_sim", global_thr["thr_sim"])),
                "thr_margin": float(v.get("thr_margin", global_thr["thr_margin"])),
                "best_score": float(v.get("best_score")) if ("best_score" in v and v.get("best_score") is not None) else None,
            }

        if getattr(config, "combine_mode_override", None) is not None:
            combine_mode = str(getattr(config, "combine_mode_override"))

        print(f"[threshold] loaded: {json_path}")
        print("[threshold] source: classwise_thresholds.json")
        print("[threshold] combine_mode:", combine_mode)
        print(f"[threshold] global: thr_sim={global_thr['thr_sim']}, thr_margin={global_thr['thr_margin']}")
        return combine_mode, global_thr, classwise

    # (B) best_2d.txt
    if bool(getattr(config, "load_threshold_from_200", False)):
        fname = str(getattr(config, "threshold_file", "best_2d.txt"))
        best_path = exp_dir / "threshold" / fname
        if best_path.exists() and threshold_source in ("auto", "best_2d", "200", "best2d"):
            kv = _parse_keyval_file(best_path)
            combine_mode = str(kv.get("combine_mode", getattr(config, "combine_mode", "or")))
            global_thr = {
                "thr_sim": float(kv.get(thr_key, kv.get("thr_median_nonzero", kv.get("thr_median", getattr(config, "unknown_threshold", 0.0))))),
                "thr_margin": float(kv.get(mthr_key, kv.get("mthr_median_nonzero", kv.get("mthr_median", getattr(config, "margin_threshold", 0.0))))),
            }
            if getattr(config, "combine_mode_override", None) is not None:
                combine_mode = str(getattr(config, "combine_mode_override"))

            print(f"[threshold] loaded: {best_path}")
            print("[threshold] source: best_2d.txt")
            print("[threshold] combine_mode:", combine_mode)
            print(f"[threshold] global: thr_sim={global_thr['thr_sim']}, thr_margin={global_thr['thr_margin']}")
            return combine_mode, global_thr, {}

    # (C) fallback
    combine_mode = str(getattr(config, "combine_mode", "or"))
    global_thr = {
        "thr_sim": float(getattr(config, "unknown_threshold", 0.0)),
        "thr_margin": float(getattr(config, "margin_threshold", 0.0)),
    }
    if getattr(config, "combine_mode_override", None) is not None:
        combine_mode = str(getattr(config, "combine_mode_override"))

    print("[threshold] fallback to config values")
    print("[threshold] combine_mode:", combine_mode)
    print(f"[threshold] global: thr_sim={global_thr['thr_sim']}, thr_margin={global_thr['thr_margin']}")
    return combine_mode, global_thr, {}


def build_threshold_tables(
    num_classes: int,
    global_thr: dict,
    classwise: dict,
    config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thr_sim_g = float(global_thr["thr_sim"])
    thr_m_g = float(global_thr["thr_margin"])

    thr_sim_table = np.full((num_classes,), thr_sim_g, dtype=np.float32)
    thr_m_table = np.full((num_classes,), thr_m_g, dtype=np.float32)
    applied = np.zeros((num_classes,), dtype=np.int8)

    use_classwise = bool(getattr(config, "use_classwise_thresholds", True))
    if (not use_classwise) or (not classwise):
        return thr_sim_table, thr_m_table, applied

    mode = str(getattr(config, "classwise_apply_mode", "all"))
    use_margin = bool(getattr(config, "classwise_use_margin", True))

    apply_classes = set(int(x) for x in getattr(config, "classwise_apply_classes", []) or [])
    exclude_classes = set(int(x) for x in getattr(config, "classwise_exclude_classes", []) or [])

    clip_sim = float(getattr(config, "classwise_sim_clip_delta", 0.0))
    clip_m = float(getattr(config, "classwise_margin_clip_delta", 0.0))

    for cid, v in classwise.items():
        cid = int(cid)
        if cid < 0 or cid >= num_classes:
            continue

        if mode == "subset" and (cid not in apply_classes):
            continue
        if mode == "exclude" and (cid in exclude_classes):
            continue
        if cid in exclude_classes:
            continue

        sim_c = float(v.get("thr_sim", thr_sim_g))
        m_c = float(v.get("thr_margin", thr_m_g))

        if clip_sim > 0:
            sim_c = float(np.clip(sim_c, thr_sim_g - clip_sim, thr_sim_g + clip_sim))
        if clip_m > 0:
            m_c = float(np.clip(m_c, thr_m_g - clip_m, thr_m_g + clip_m))

        if mode == "looser_sim_only":
            thr_sim_table[cid] = min(thr_sim_g, sim_c)
            thr_m_table[cid] = thr_m_g
            applied[cid] = 1
        elif mode == "looser_only":
            thr_sim_table[cid] = min(thr_sim_g, sim_c)
            thr_m_table[cid] = (min(thr_m_g, m_c) if use_margin else thr_m_g)
            applied[cid] = 1
        else:
            thr_sim_table[cid] = sim_c
            thr_m_table[cid] = (m_c if use_margin else thr_m_g)
            applied[cid] = 1

    for cid in exclude_classes:
        if 0 <= cid < num_classes:
            thr_sim_table[cid] = thr_sim_g
            thr_m_table[cid] = thr_m_g
            applied[cid] = 0

    return thr_sim_table, thr_m_table, applied


# ===================================
# multi-fold helpers
# ===================================
def _strip_fold_suffix(exp: str) -> str:
    return re.sub(r"_fold\d+$", "", str(exp))


def _infer_name_with_debug(config) -> str:
    infer_name = str(getattr(config, "infer_name", "300_infer"))
    if bool(getattr(config, "debug", False)):
        infer_name = f"{infer_name}_debug"
    return infer_name


def apply_unknown_from_sims(
    sims: np.ndarray,
    thr_sim_table: np.ndarray,
    thr_m_table: np.ndarray,
    combine_mode: str,
    config,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    top2_idx = np.argpartition(-sims, kth=1, axis=1)[:, :2]
    top2_val = np.take_along_axis(sims, top2_idx, axis=1)
    order = np.argsort(-top2_val, axis=1)

    top1 = top2_val[np.arange(len(sims)), order[:, 0]]
    top2 = top2_val[np.arange(len(sims)), order[:, 1]]
    pred = top2_idx[np.arange(len(sims)), order[:, 0]].astype(int)
    margin = (top1 - top2).astype(np.float32)

    thr_sim_used = thr_sim_table[pred]
    thr_m_used = thr_m_table[pred]

    unknown_by_sim = top1 < thr_sim_used

    gate_sim = getattr(config, "margin_gate_sim", None)
    gate_add = float(getattr(config, "margin_gate_add", 0.0))
    if gate_sim is not None:
        gate_mask = top1 < float(gate_sim)
    elif gate_add > 0:
        gate_mask = top1 < (thr_sim_used + gate_add)
    else:
        gate_mask = np.ones_like(top1, dtype=bool)

    unknown_by_margin = gate_mask & (margin < thr_m_used)

    if combine_mode == "or":
        unknown = unknown_by_sim | unknown_by_margin
    elif combine_mode == "and":
        unknown = unknown_by_sim & unknown_by_margin
    else:
        raise ValueError(f"unknown combine_mode: {combine_mode}")

    pred_out = np.where(unknown, -1, pred).astype(int)

    dbg = pd.DataFrame({
        "max_sim": top1.astype(float),
        "second_sim": top2.astype(float),
        "margin": margin.astype(float),
        "pred_raw": pred.astype(int),
        "thr_sim_used": thr_sim_used.astype(float),
        "thr_margin_used": thr_m_used.astype(float),
        "gate_mask": gate_mask.astype(int),
        "unknown_by_sim": unknown_by_sim.astype(int),
        "unknown_by_margin": unknown_by_margin.astype(int),
        "unknown": unknown.astype(int),
        "pred": pred_out.astype(int),
    })

    summary = (
        dbg.groupby("pred_raw")
           .agg(
               n=("pred_raw", "size"),
               unknown_rate=("unknown", "mean"),
               mean_sim=("max_sim", "mean"),
               mean_margin=("margin", "mean"),
               thr_sim_mean=("thr_sim_used", "mean"),
               thr_m_mean=("thr_margin_used", "mean"),
           )
           .reset_index()
           .sort_values("pred_raw")
    )
    return pred_out, dbg, summary


def run_one_exp(train_exp: str, infer_cfg: dict) -> dict:
    train_cfg = load_train_yaml(infer_cfg["train_output_dir"], train_exp)
    if isinstance(train_cfg, dict) and "100_train_arcface" in train_cfg:
        train_cfg = train_cfg["100_train_arcface"]

    merged = OmegaConf.merge(OmegaConf.create(train_cfg), OmegaConf.create(infer_cfg))
    config = dict_to_namespace(OmegaConf.to_container(merged, resolve=True))
    infer_name = _infer_name_with_debug(config)

    exp_dir = Path(config.train_output_dir) / train_exp
    outdir = exp_dir / "inference" / infer_name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "yaml").mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, outdir / "yaml" / "300_config.yaml")

    print("\n==============================")
    print(f"[300] train_exp = {train_exp}")
    print("==============================")

    combine_mode, global_thr, classwise = load_thresholds(exp_dir, config)
    thr_sim_table, thr_m_table, applied_table = build_threshold_tables(
        num_classes=int(config.num_classes),
        global_thr=global_thr,
        classwise=classwise,
        config=config,
    )

    pd.DataFrame({
        "class_id": np.arange(int(config.num_classes)),
        "thr_sim": thr_sim_table,
        "thr_margin": thr_m_table,
        "classwise_applied": applied_table.astype(int),
    }).to_csv(outdir / "threshold_table.csv", index=False)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    use_clean = bool(getattr(config, "use_clean_train_csv", True))
    train_csv_clean = exp_dir / "train_meta_pp_clean.csv"
    if use_clean and train_csv_clean.exists():
        train_csv = train_csv_clean
    else:
        if hasattr(config, "train_pp_csv") and str(getattr(config, "train_pp_csv")):
            train_csv = Path(str(getattr(config, "train_pp_csv")))
        else:
            train_csv = Path(config.pp_dir) / config.pp_exp / "train_meta_pp.csv"

    test_csv = Path(config.pp_dir) / config.pp_exp / "test_meta_pp.csv"
    print(f"[csv] train_csv: {train_csv}")
    print(f"[csv] test_csv : {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if "is_junk" in train_df.columns:
        before = len(train_df)
        train_df = train_df[train_df["is_junk"].fillna(0).astype(int) == 0].reset_index(drop=True)
        print(f"[train_df] drop is_junk: {before} -> {len(train_df)}")

    num_features = list(config.num_features) if bool(getattr(config, "use_tabular", True)) else []
    train_df = prepare_df(train_df, num_features)
    test_df = prepare_df(test_df, num_features)

    tf = get_val_transform(config.img_size)

    train_ds = PlayerDataset(df=train_df, transform=tf, num_features=num_features, is_train=False, p_hflip=0.0, cache_images=False)
    test_ds = PlayerDataset(df=test_df, transform=tf, num_features=num_features, is_train=False, p_hflip=0.0, cache_images=False)

    train_loader = DataLoader(train_ds, batch_size=int(config.batch_size), shuffle=False, num_workers=int(config.num_workers), pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=int(config.batch_size), shuffle=False, num_workers=int(config.num_workers), pin_memory=True, drop_last=False)

    ckpt_path = exp_dir / "model" / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best.ckpt not found: {ckpt_path}")

    module = PlayerLightningModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model_name=config.model_name,
        num_classes=int(config.num_classes),
        num_tabular_features=len(num_features) if bool(getattr(config, "use_tabular", True)) else 0,
        embedding_dim=int(config.embedding_dim),
        pretrained=False,
        arcface_s=float(config.arcface_s),
        arcface_m=float(config.arcface_m),
        lr=1e-3,
        weight_decay=0.0,
        epochs=1,
        use_ema=bool(getattr(config, "use_ema", False)),
        ema_decay=float(getattr(config, "ema_decay", 0.0)),
    ).to(device)
    module.eval()
    module.setup("fit")

    col2idx = {c: i for i, c in enumerate(num_features)}

    print("computing prototypes...")
    prototypes = compute_prototypes(module, train_loader, device, num_classes=int(config.num_classes))
    np.save(outdir / "prototypes.npy", prototypes.numpy())

    print("extracting test embeddings...")
    test_emb = extract_embeddings(module, test_loader, device, tta_hflip=bool(getattr(config, "tta_hflip", True)), col2idx=col2idx)
    np.save(outdir / "test_emb.npy", test_emb.numpy())

    sims = (test_emb.numpy() @ prototypes.numpy().T).astype(np.float32)

    pred_out, dbg, summary = apply_unknown_from_sims(
        sims=sims,
        thr_sim_table=thr_sim_table,
        thr_m_table=thr_m_table,
        combine_mode=combine_mode,
        config=config,
    )

    pd.DataFrame({"label_id": pred_out}).to_csv(outdir / "submission.csv", index=False)
    dbg.to_csv(outdir / "test_pred_debug.csv", index=False)
    summary.to_csv(outdir / "pred_class_summary.csv", index=False)

    print("saved:", outdir)
    print("pred distribution:\n", pd.Series(pred_out).value_counts().sort_index())
    print("[unknown] rate:", float((pred_out == -1).mean()))

    # cleanup
    del module, prototypes, test_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "train_exp": train_exp,
        "exp_dir": exp_dir,
        "outdir": outdir,
        "pred_out": pred_out,
        "sims": sims,
        "thr_sim_table": thr_sim_table.astype(np.float32),
        "thr_m_table": thr_m_table.astype(np.float32),
        "combine_mode": combine_mode,
        "config": config,
    }

def _norm_pos(pos: np.ndarray, mask_real: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """
    pos: (T,B,2)
    mask_real: (T,B) True for real bbox (not padded)
    """
    if mode == "none":
        return pos

    real = pos[mask_real]
    if real.size == 0:
        return pos

    if mode == "minmax":
        pmin = real.min(axis=0)
        pmax = real.max(axis=0)
        denom = np.where(pmax > pmin, pmax - pmin, 1.0)
        return (pos - pmin) / denom

    raise ValueError(f"unknown ilp_pos_norm: {mode}")


def post_processing_ilp(
    probs: np.ndarray,      # (T,B,P)
    bbox_pos: np.ndarray,   # (T,B,2)
    mu: float = 2.0,
    lambda_1: float = 1.0,
    lambda_2: float = 0.5,
    MAX_DIST: float = 1.0,
    max_players: int = 10,
    time_limit: int = 0,
):
    """
    ILP assignment.
    label index:
      0 = NO_PERSON (=> -1)
      1..P-1 = real players (=> label_id = p-1)
    """
    import pulp  # 遅延 import

    NO_PERSON = 0
    T, B, P = probs.shape
    REAL_PLAYERS = range(1, P)

    problem = pulp.LpProblem("Player_Assignment", pulp.LpMaximize)

    # Variables
    x = pulp.LpVariable.dicts(
        "x",
        ((t, b, p) for t in range(T) for b in range(B) for p in range(P)),
        cat="Binary"
    )

    y = pulp.LpVariable.dicts(
        "y",
        (p for p in REAL_PLAYERS),
        cat="Binary"
    )

    z1 = pulp.LpVariable.dicts(
        "z1",
        ((t, b1, b2, p)
         for t in range(1, T)
         for b1 in range(B)
         for b2 in range(B)
         for p in REAL_PLAYERS),
        cat="Binary"
    )

    z2 = pulp.LpVariable.dicts(
        "z2",
        ((t, b1, b2, p)
         for t in range(2, T)
         for b1 in range(B)
         for b2 in range(B)
         for p in REAL_PLAYERS),
        cat="Binary"
    )

    w = pulp.LpVariable.dicts(
        "w",
        ((t, b1, b2, p)
         for t in range(T)
         for b1 in range(B)
         for b2 in range(b1 + 1, B)
         for p in REAL_PLAYERS),
        cat="Binary"
    )

    # Distances
    dist_1 = {}
    dist_2 = {}
    dist_same = {}

    for t in range(1, T):
        for b1 in range(B):
            for b2 in range(B):
                dist_1[(t, b1, b2)] = float(np.linalg.norm(bbox_pos[t - 1, b1] - bbox_pos[t, b2]))

    for t in range(2, T):
        for b1 in range(B):
            for b2 in range(B):
                dist_2[(t, b1, b2)] = float(np.linalg.norm(bbox_pos[t - 2, b1] - bbox_pos[t, b2]))

    for t in range(T):
        for b1 in range(B):
            for b2 in range(b1 + 1, B):
                dist_same[(t, b1, b2)] = float(np.linalg.norm(bbox_pos[t, b1] - bbox_pos[t, b2]))

    # Objective
    problem += (
        pulp.lpSum(
            float(probs[t, b, p]) * x[(t, b, p)]
            for t in range(T) for b in range(B) for p in range(P)
        )
        - float(lambda_1) * pulp.lpSum(
            dist_1[(t, b1, b2)] * z1[(t, b1, b2, p)]
            for (t, b1, b2, p) in z1
        )
        - float(lambda_2) * pulp.lpSum(
            dist_2[(t, b1, b2)] * z2[(t, b1, b2, p)]
            for (t, b1, b2, p) in z2
        )
        - float(mu) * pulp.lpSum(
            dist_same[(t, b1, b2)] * w[(t, b1, b2, p)]
            for (t, b1, b2, p) in w
        )
    )

    # Constraints
    # (1) each bbox exactly one label
    for t in range(T):
        for b in range(B):
            problem += pulp.lpSum(x[(t, b, p)] for p in range(P)) == 1

    # (2) x -> y (exclude NO_PERSON)
    for t in range(T):
        for b in range(B):
            for p in REAL_PLAYERS:
                problem += x[(t, b, p)] <= y[p]

    # (3) total number of players <= max_players
    problem += pulp.lpSum(y[p] for p in REAL_PLAYERS) <= int(max_players)

    # (4) transition consistency (t-1)
    for (t, b1, b2, p) in z1:
        problem += z1[(t, b1, b2, p)] <= x[(t - 1, b1, p)]
        problem += z1[(t, b1, b2, p)] <= x[(t, b2, p)]
        problem += z1[(t, b1, b2, p)] >= x[(t - 1, b1, p)] + x[(t, b2, p)] - 1

        if dist_1[(t, b1, b2)] > MAX_DIST:
            problem += z1[(t, b1, b2, p)] == 0

    # (5) transition consistency (t-2)
    for (t, b1, b2, p) in z2:
        problem += z2[(t, b1, b2, p)] <= x[(t - 2, b1, p)]
        problem += z2[(t, b1, b2, p)] <= x[(t, b2, p)]
        problem += z2[(t, b1, b2, p)] >= x[(t - 2, b1, p)] + x[(t, b2, p)] - 1

        if dist_2[(t, b1, b2)] > MAX_DIST:
            problem += z2[(t, b1, b2, p)] == 0

    # (6) same-time duplicate
    for (t, b1, b2, p) in w:
        problem += w[(t, b1, b2, p)] <= x[(t, b1, p)]
        problem += w[(t, b1, b2, p)] <= x[(t, b2, p)]
        problem += w[(t, b1, b2, p)] >= x[(t, b1, p)] + x[(t, b2, p)] - 1

        if dist_same[(t, b1, b2)] > MAX_DIST:
            problem += w[(t, b1, b2, p)] == 0

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    if int(time_limit) > 0:
        # pulpのバージョンによって引数名が違うことがあるので安全に options を使う
        solver = pulp.PULP_CBC_CMD(msg=False, options=[f"sec {int(time_limit)}"])

    problem.solve(solver)

    assignment = np.full((T, B), 0, dtype=np.int32)
    for t in range(T):
        for b in range(B):
            for p in range(P):
                if pulp.value(x[(t, b, p)]) > 0.5:
                    assignment[t, b] = int(p)
                    break

    return assignment


def ilp_postprocess_dataframe(
    test_df: pd.DataFrame,
    sims: np.ndarray,                # (N,C) C=num_classes
    config,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    test_df の index と sims の行が一致している前提（あなたのコードは一致します）。
    戻り:
      pred_out: (N,) label_id in {-1,0..C-1}
      dbg_df: 主要統計
    """
    group_cols = list(getattr(config, "ilp_group_cols", ["quarter", "session", "angle"]))
    frame_col = str(getattr(config, "ilp_frame_col", "frame"))
    pos_cols = list(getattr(config, "ilp_pos_cols", ["fx", "fy"]))
    pos_norm = str(getattr(config, "ilp_pos_norm", "minmax"))

    unknown_score = float(getattr(config, "ilp_unknown_score", getattr(config, "unknown_threshold", 0.77)))
    score_scale = float(getattr(config, "ilp_score_scale", 10.0))

    mu = float(getattr(config, "ilp_mu", 2.0))
    lam1 = float(getattr(config, "ilp_lambda_1", 1.0))
    lam2 = float(getattr(config, "ilp_lambda_2", 0.5))
    max_dist = float(getattr(config, "ilp_max_dist", 1.0))
    max_players = int(getattr(config, "ilp_max_players", 10))
    time_limit = int(getattr(config, "ilp_time_limit", 0))

    N, C = sims.shape
    P = C + 1  # + NO_PERSON

    pred_out = np.full((N,), -1, dtype=np.int32)

    dbg_rows = []
    gobj = test_df.groupby(group_cols, sort=False)

    for gkey, gdf in gobj:
        # frameで時系列順
        frames = []
        frame_indices = []
        for f, fdf in gdf.groupby(frame_col, sort=True):
            frames.append(f)
            frame_indices.append(fdf.index.to_numpy())

        T = len(frames)
        if T == 0:
            continue

        B = max(len(ix) for ix in frame_indices)
        if B == 0:
            continue

        # probs / pos 作成（paddingあり）
        probs = np.full((T, B, P), -1e9, dtype=np.float32)
        probs[:, :, 0] = unknown_score * score_scale  # NO_PERSON の定数スコア

        pos = np.zeros((T, B, 2), dtype=np.float32)
        mask_real = np.zeros((T, B), dtype=bool)

        for t, idxs in enumerate(frame_indices):
            K = len(idxs)
            if K == 0:
                continue
            mask_real[t, :K] = True

            # 位置
            pos[t, :K, 0] = test_df.loc[idxs, pos_cols[0]].to_numpy(np.float32)
            pos[t, :K, 1] = test_df.loc[idxs, pos_cols[1]].to_numpy(np.float32)

            # スコア（1..C）
            probs[t, :K, 1:] = (sims[idxs].astype(np.float32) * score_scale)

        pos = _norm_pos(pos, mask_real, mode=pos_norm)

        assign = post_processing_ilp(
            probs=probs,
            bbox_pos=pos,
            mu=mu,
            lambda_1=lam1,
            lambda_2=lam2,
            MAX_DIST=max_dist,
            max_players=max_players,
            time_limit=time_limit,
        )  # (T,B) int

        # assignment -> label_id
        # p=0 => -1
        # p>=1 => p-1 (0..C-1)
        for t, idxs in enumerate(frame_indices):
            K = len(idxs)
            if K == 0:
                continue
            a = assign[t, :K]
            y = np.where(a == 0, -1, a - 1).astype(np.int32)
            pred_out[idxs] = y

        unk_rate = float((np.where(assign[:, :], assign, 0) == 0).mean())
        dbg_rows.append({
            "group": str(gkey),
            "T": int(T),
            "B": int(B),
            "unknown_rate_including_pad": unk_rate,
        })

    dbg_df = pd.DataFrame(dbg_rows)
    return pred_out, dbg_df

# ===================================
# main (multi-fold)
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    infer_cfg = OmegaConf.to_container(cfg["300_infer"], resolve=True)

    folds = list(infer_cfg.get("folds", [])) if isinstance(infer_cfg, dict) else []
    base_exp = _strip_fold_suffix(str(infer_cfg["exp"]))
    if folds:
        train_exps = [f"{base_exp}_fold{int(f)}" for f in folds]
    else:
        train_exps = [str(infer_cfg["exp"])]

    print(f"[300] base_exp={base_exp}")
    print(f"[300] train_exps={train_exps}")

    results = []
    sims_sum = None
    thr_sim_tables = []
    thr_m_tables = []
    combine_modes = []

    for train_exp in train_exps:
        res = run_one_exp(train_exp=train_exp, infer_cfg=infer_cfg)
        results.append(res)
        combine_modes.append(res["combine_mode"])
        thr_sim_tables.append(res["thr_sim_table"])
        thr_m_tables.append(res["thr_m_table"])

        if folds:
            sims_sum = res["sims"].astype(np.float64) if sims_sum is None else (sims_sum + res["sims"].astype(np.float64))

    # ensemble
    ensemble_enable = bool(infer_cfg.get("ensemble_enable", True))
    if folds and ensemble_enable and (sims_sum is not None) and (len(results) >= 2):
        combine_mode_ens = combine_modes[0] if len(set(combine_modes)) == 1 else str(infer_cfg.get("combine_mode", "or"))
        sims_mean = (sims_sum / float(len(results))).astype(np.float32)

        thr_sim_ens = np.median(np.stack(thr_sim_tables, axis=0), axis=0).astype(np.float32)
        thr_m_ens = np.median(np.stack(thr_m_tables, axis=0), axis=0).astype(np.float32)

        ens_suffix = str(infer_cfg.get("ensemble_exp_suffix", "_ens"))
        ens_exp = f"{base_exp}{ens_suffix}"
        ens_exp_dir = Path(infer_cfg["train_output_dir"]) / ens_exp

        infer_name = str(infer_cfg.get("infer_name", "300_infer"))
        if bool(infer_cfg.get("debug", False)):
            infer_name = f"{infer_name}_debug"

        outdir = ens_exp_dir / "inference" / infer_name
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "yaml").mkdir(parents=True, exist_ok=True)
        save_config_yaml(dict_to_namespace(infer_cfg), outdir / "yaml" / "300_ensemble_config.yaml")

        pd.DataFrame({
            "class_id": np.arange(len(thr_sim_ens)),
            "thr_sim": thr_sim_ens,
            "thr_margin": thr_m_ens,
        }).to_csv(outdir / "threshold_table_ens.csv", index=False)

        base_config = results[0]["config"]
        pred_out, dbg, summary = apply_unknown_from_sims(
            sims=sims_mean,
            thr_sim_table=thr_sim_ens,
            thr_m_table=thr_m_ens,
            combine_mode=combine_mode_ens,
            config=base_config,
        )

        pd.DataFrame({"label_id": pred_out}).to_csv(outdir / "submission.csv", index=False)
        dbg.to_csv(outdir / "test_pred_debug.csv", index=False)
        summary.to_csv(outdir / "pred_class_summary.csv", index=False)
        np.save(outdir / "sims_mean.npy", sims_mean)

        meta = {
            "base_exp": base_exp,
            "train_exps": train_exps,
            "combine_mode": combine_mode_ens,
            "n_folds": len(results),
        }
        (outdir / "ensemble_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        print("\n==============================")
        print("[ENSEMBLE] saved:", outdir)
        print("pred distribution:\n", pd.Series(pred_out).value_counts().sort_index())
        print("[unknown] rate:", float((pred_out == -1).mean()))
        print("==============================\n")

        # ★追加：ILP版（submission_ilp.csv）
        if bool(infer_cfg.get("ilp_enable", False)):
            # test_df を読み直して同じ前処理
            test_csv = Path(infer_cfg["pp_dir"]) / infer_cfg["pp_exp"] / "test_meta_pp.csv"
            test_df = pd.read_csv(test_csv)

            num_features = list(base_config.num_features) if bool(getattr(base_config, "use_tabular", True)) else []
            test_df = prepare_df(test_df, num_features)

            pred_ilp, ilp_dbg = ilp_postprocess_dataframe(
                test_df=test_df,
                sims=sims_mean,
                config=base_config,
            )

            pd.DataFrame({"label_id": pred_ilp}).to_csv(outdir / "submission_ilp.csv", index=False)
            ilp_dbg.to_csv(outdir / "ilp_group_debug.csv", index=False)

            print("[ILP] unknown rate:", float((pred_ilp == -1).mean()))


if __name__ == "__main__":
    main()