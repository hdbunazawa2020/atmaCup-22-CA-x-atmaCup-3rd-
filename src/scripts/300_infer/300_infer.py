import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append("..")  # src/scripts から実行する想定

from utils.data import save_config_yaml, dict_to_namespace
from atma_datasets.player_dataset import PlayerDataset
from atma_datasets.transforms import get_val_transform
from models.lightning_module import PlayerLightningModule

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")

# ===================================
# utils
# ===================================
def load_train_yaml(train_output_dir: str, exp: str) -> dict:
    p = Path(train_output_dir) / exp / "yaml" / "config.yaml"
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

def flip_num_feats_batch(x: torch.Tensor, n_players: torch.Tensor, col2idx: dict[str,int]) -> torch.Tensor:
    x = x.clone()
    if "fx" in col2idx:
        x[:, col2idx["fx"]] = -x[:, col2idx["fx"]]
    if "rank_x" in col2idx:
        rx = x[:, col2idx["rank_x"]]
        x[:, col2idx["rank_x"]] = (n_players.float() - 1.0) - rx
    return x

@torch.no_grad()
def compute_prototypes(module: PlayerLightningModule, loader: DataLoader, device, num_classes: int) -> torch.Tensor:
    module.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["num"].to(device, non_blocking=True)
        y   = batch["label"].to(device, non_blocking=True)

        if module.use_ema and module.model_ema is not None:
            emb = module.model_ema.module.get_embedding(img, tab)
        else:
            emb = module.model.get_embedding(img, tab)

        emb = F.normalize(emb, p=2, dim=1)

        if sums is None:
            sums = torch.zeros(num_classes, emb.size(1), device=device, dtype=emb.dtype)

        for c in range(num_classes):
            m = (y == c)
            if m.any():
                sums[c] += emb[m].sum(dim=0)
                counts[c] += m.sum().float()

    prot = sums / counts.clamp(min=1.0).unsqueeze(1)
    prot = F.normalize(prot, p=2, dim=1)
    return prot.detach().cpu()

@torch.no_grad()
def extract_embeddings(module: PlayerLightningModule, loader: DataLoader, device,
                       tta_hflip: bool, col2idx: dict[str,int]) -> torch.Tensor:
    module.eval()
    embs = []

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["num"].to(device, non_blocking=True)
        n_players = batch.get("n_players", None)
        if n_players is None:
            n_players = torch.full((img.size(0),), 10, device=device, dtype=torch.long)
        else:
            n_players = n_players.to(device, non_blocking=True)

        if module.use_ema and module.model_ema is not None:
            e1 = module.model_ema.module.get_embedding(img, tab)
        else:
            e1 = module.model.get_embedding(img, tab)
        e1 = F.normalize(e1, p=2, dim=1)

        if not tta_hflip:
            embs.append(e1.cpu())
            continue

        img_f = torch.flip(img, dims=[3])
        tab_f = flip_num_feats_batch(tab, n_players, col2idx)

        if module.use_ema and module.model_ema is not None:
            e2 = module.model_ema.module.get_embedding(img_f, tab_f)
        else:
            e2 = module.model.get_embedding(img_f, tab_f)
        e2 = F.normalize(e2, p=2, dim=1)

        e = F.normalize((e1 + e2) * 0.5, p=2, dim=1)
        embs.append(e.cpu())

    return torch.cat(embs, dim=0)

def load_thresholds(exp_dir: Path) -> tuple[str, dict, dict]:
    """
    Returns:
      combine_mode: "or" or "and"
      global_thr: {"thr_sim": float, "thr_margin": float}
      classwise: {int: {"thr_sim": float, "thr_margin": float}}
    """
    json_path = exp_dir / "threshold" / "classwise_thresholds.json"
    if not json_path.exists():
        raise FileNotFoundError(f"classwise_thresholds.json not found: {json_path}")

    with open(json_path, "r") as f:
        payload = json.load(f)

    combine_mode = str(payload.get("combine_mode", "or"))
    g = payload.get("global", {})
    global_thr = {
        "thr_sim": float(g.get("thr_median_nonzero", g.get("thr_median", 0.0))),
        "thr_margin": float(g.get("mthr_median_nonzero", g.get("mthr_median", 0.0))),
    }

    cw_raw = payload.get("classwise", {})
    classwise = {}
    for k, v in cw_raw.items():
        try:
            cid = int(k)
        except Exception:
            continue
        classwise[cid] = {
            "thr_sim": float(v.get("thr_sim", global_thr["thr_sim"])),
            "thr_margin": float(v.get("thr_margin", global_thr["thr_margin"])),
        }

    print(f"[threshold] loaded: {json_path}")
    print(f"[threshold] combine_mode: {combine_mode}")
    print(f"[threshold] global (fallback): thr_sim={global_thr['thr_sim']}, thr_margin={global_thr['thr_margin']}")
    return combine_mode, global_thr, classwise

# ===================================
# main
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # merge train cfg + infer cfg
    infer_cfg = OmegaConf.to_container(cfg["300_infer"], resolve=True)
    train_cfg = load_train_yaml(infer_cfg["train_output_dir"], infer_cfg["exp"])
    if "100_train_arcface" in train_cfg:
        train_cfg = train_cfg["100_train_arcface"]
    merged = OmegaConf.merge(OmegaConf.create(train_cfg), OmegaConf.create(infer_cfg))
    config = dict_to_namespace(OmegaConf.to_container(merged, resolve=True))

    if config.debug:
        config.exp = "300_infer_debug"

    exp_dir = Path(config.train_output_dir) / str(config.exp)
    outdir = exp_dir / "inference" / "300_infer"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "yaml").mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, outdir / "yaml" / "300_config.yaml")

    # thresholds
    combine_mode, global_thr, classwise = load_thresholds(exp_dir)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # csv paths
    train_csv = exp_dir / "train_meta_pp_clean.csv"
    test_csv  = Path(config.pp_dir) / config.pp_exp / "test_meta_pp.csv"
    print(f"[csv] train_csv: {train_csv}")
    print(f"[csv] test_csv : {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # drop junk if exists
    if "is_junk" in train_df.columns:
        before = len(train_df)
        train_df = train_df[train_df["is_junk"].fillna(0).astype(int) == 0].reset_index(drop=True)
        print(f"[train_df] drop is_junk: {before} -> {len(train_df)}")

    num_features = list(config.num_features) if config.use_tabular else []
    train_df = prepare_df(train_df, num_features)
    test_df  = prepare_df(test_df, num_features)

    tf = get_val_transform(config.img_size)

    train_ds = PlayerDataset(
        df=train_df,
        transform=tf,
        num_features=num_features,
        is_train=False,
        p_hflip=0.0,
        cache_images=False,
    )
    test_ds = PlayerDataset(
        df=test_df,
        transform=tf,
        num_features=num_features,
        is_train=False,
        p_hflip=0.0,
        cache_images=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False
    )

    # load module
    ckpt_path = exp_dir / "model" / "best.ckpt"
    module = PlayerLightningModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model_name=config.model_name,
        num_classes=config.num_classes,
        num_tabular_features=len(num_features) if config.use_tabular else 0,
        embedding_dim=config.embedding_dim,
        pretrained=False,
        arcface_s=config.arcface_s,
        arcface_m=config.arcface_m,
        lr=1e-3,
        weight_decay=0.0,
        epochs=1,
        use_ema=config.use_ema,
        ema_decay=config.ema_decay,
    )
    module = module.to(device)
    module.eval()
    module.setup("fit")

    col2idx = {c:i for i,c in enumerate(num_features)}

    # prototypes
    print("computing prototypes...")
    prototypes = compute_prototypes(module, train_loader, device, num_classes=config.num_classes)
    np.save(outdir / "prototypes.npy", prototypes.numpy())

    # embeddings
    print("extracting test embeddings...")
    test_emb = extract_embeddings(
        module, test_loader, device,
        tta_hflip=bool(config.tta_hflip),
        col2idx=col2idx
    )
    np.save(outdir / "test_emb.npy", test_emb.numpy())

    # cosine similarity
    sims = test_emb.numpy() @ prototypes.numpy().T  # (N, C)

    # top-1 / top-2
    top2_idx = np.argpartition(-sims, kth=1, axis=1)[:, :2]
    top2_val = np.take_along_axis(sims, top2_idx, axis=1)
    order = np.argsort(-top2_val, axis=1)

    top1 = top2_val[np.arange(len(sims)), order[:, 0]]
    top2 = top2_val[np.arange(len(sims)), order[:, 1]]
    pred = top2_idx[np.arange(len(sims)), order[:, 0]]
    margin = top1 - top2

    # ---- class-wise thresholding ----
    thr_sim_used = np.zeros_like(top1, dtype=np.float32)
    thr_m_used   = np.zeros_like(top1, dtype=np.float32)

    for i, c in enumerate(pred.astype(int)):
        if int(c) in classwise:
            thr_sim_used[i] = float(classwise[int(c)]["thr_sim"])
            thr_m_used[i]   = float(classwise[int(c)]["thr_margin"])
        else:
            thr_sim_used[i] = float(global_thr["thr_sim"])
            thr_m_used[i]   = float(global_thr["thr_margin"])

    unknown_by_sim = top1 < thr_sim_used
    unknown_by_m   = margin < thr_m_used

    if combine_mode == "or":
        unknown = unknown_by_sim | unknown_by_m
    elif combine_mode == "and":
        unknown = unknown_by_sim & unknown_by_m
    else:
        raise ValueError(f"unknown combine_mode: {combine_mode}")

    pred_out = np.where(unknown, -1, pred).astype(int)

    # submission
    sub = pd.DataFrame({"label_id": pred_out})
    sub.to_csv(outdir / "submission.csv", index=False)

    dbg = pd.DataFrame({
        "max_sim": top1,
        "second_sim": top2,
        "margin": margin,
        "pred_raw": pred.astype(int),
        "thr_sim_used": thr_sim_used,
        "thr_margin_used": thr_m_used,
        "unknown_by_sim": unknown_by_sim.astype(int),
        "unknown_by_margin": unknown_by_m.astype(int),
        "unknown": unknown.astype(int),
        "pred": pred_out,
    })
    dbg.to_csv(outdir / "test_pred_debug.csv", index=False)

    print("saved:", outdir)
    print("pred distribution:\n", pd.Series(pred_out).value_counts().sort_index())
    print("[unknown] rate:", float((pred_out == -1).mean()))

if __name__ == "__main__":
    main()