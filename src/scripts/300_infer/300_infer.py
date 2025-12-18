import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any

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
    """
    - angle_id
    - n_players (quarter, session, frame, angle ごと)
    - num_features の NaN 埋め
    """
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
    """
    画像 hflip に合わせた tabular の左右反転補正
    - fx: 符号反転
    - rank_x: n-1-rank_x
    - rank_x_norm: 1-rank_x_norm (n<=1 のときは 0)
    """
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
    """
    train 全量から class prototype を作る（推論寄り）
    - EMAがあればEMA
    - L2 normalize
    - index_add で高速化
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
    """
    test embedding 抽出（必要ならhflip TTA）
    """
    module.eval()
    embs = []

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["num"].to(device, non_blocking=True)

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
            continue

        # ---- flip ----
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
    """
    best_2d.txt などの key=value を読む
    """
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
        # try float
        try:
            kv[k] = float(v)
        except Exception:
            kv[k] = v
    return kv


def load_thresholds(exp_dir: Path, config) -> tuple[str, dict, dict]:
    """
    優先順位（default: threshold_source=auto）
      1) classwise_thresholds.json（存在すれば）
      2) best_2d.txt（load_threshold_from_200=true & 存在すれば）
      3) config の unknown_threshold / margin_threshold

    Returns:
      combine_mode: "or" or "and"
      global_thr: {"thr_sim": float, "thr_margin": float}
      classwise: {int: {"thr_sim": float, "thr_margin": float, "best_score": float|None}}
    """
    threshold_source = str(getattr(config, "threshold_source", "auto"))

    # config keys
    thr_key = str(getattr(config, "thr_key", "thr_median_nonzero"))
    mthr_key = str(getattr(config, "mthr_key", "mthr_median_nonzero"))

    # ---- (A) classwise json ----
    json_path = exp_dir / "threshold" / "classwise_thresholds.json"
    if threshold_source in ("auto", "classwise", "classwise_json") and json_path.exists():
        with open(json_path, "r") as f:
            payload = json.load(f)

        combine_mode = str(payload.get("combine_mode", getattr(config, "combine_mode", "or")))

        g = payload.get("global", {})
        global_thr = {
            "thr_sim": float(
                g.get(thr_key, g.get("thr_median_nonzero", g.get("thr_median", getattr(config, "unknown_threshold", 0.0))))
            ),
            "thr_margin": float(
                g.get(mthr_key, g.get("mthr_median_nonzero", g.get("mthr_median", getattr(config, "margin_threshold", 0.0))))
            ),
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
                "best_score": float(v.get("best_score")) if ("best_score" in v and v.get("best_score") is not None) else None,
            }

        # override combine_mode if explicitly set
        if getattr(config, "combine_mode_override", None) is not None:
            combine_mode = str(getattr(config, "combine_mode_override"))

        print(f"[threshold] loaded: {json_path}")
        print(f"[threshold] source: classwise_thresholds.json")
        print(f"[threshold] combine_mode: {combine_mode}")
        print(f"[threshold] global (fallback): thr_sim={global_thr['thr_sim']}, thr_margin={global_thr['thr_margin']}")
        return combine_mode, global_thr, classwise

    # ---- (B) best_2d.txt ----
    if bool(getattr(config, "load_threshold_from_200", False)):
        fname = str(getattr(config, "threshold_file", "best_2d.txt"))
        best_path = exp_dir / "threshold" / fname
        if best_path.exists() and threshold_source in ("auto", "best_2d", "200", "best2d"):
            kv = _parse_keyval_file(best_path)
            combine_mode = str(kv.get("combine_mode", getattr(config, "combine_mode", "or")))
            global_thr = {
                "thr_sim": float(
                    kv.get(thr_key, kv.get("thr_median_nonzero", kv.get("thr_median", getattr(config, "unknown_threshold", 0.0))))
                ),
                "thr_margin": float(
                    kv.get(mthr_key, kv.get("mthr_median_nonzero", kv.get("mthr_median", getattr(config, "margin_threshold", 0.0))))
                ),
            }
            if getattr(config, "combine_mode_override", None) is not None:
                combine_mode = str(getattr(config, "combine_mode_override"))

            print(f"[threshold] loaded: {best_path}")
            print(f"[threshold] source: best_2d.txt")
            print(f"[threshold] combine_mode: {combine_mode}")
            print(f"[threshold] global: thr_sim={global_thr['thr_sim']}, thr_margin={global_thr['thr_margin']}")
            return combine_mode, global_thr, {}

    # ---- (C) fallback ----
    combine_mode = str(getattr(config, "combine_mode", "or"))
    global_thr = {
        "thr_sim": float(getattr(config, "unknown_threshold", 0.0)),
        "thr_margin": float(getattr(config, "margin_threshold", 0.0)),
    }
    if getattr(config, "combine_mode_override", None) is not None:
        combine_mode = str(getattr(config, "combine_mode_override"))

    print("[threshold] fallback to config values")
    print(f"[threshold] combine_mode: {combine_mode}")
    print(f"[threshold] global: thr_sim={global_thr['thr_sim']}, thr_margin={global_thr['thr_margin']}")
    return combine_mode, global_thr, {}


def build_threshold_tables(
    num_classes: int,
    global_thr: dict,
    classwise: dict,
    config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pred_class(0..C-1) から O(1) で閾値を引けるように table 化

    Returns:
      thr_sim_table: (C,)
      thr_margin_table: (C,)
      applied_table: (C,) 0/1 (classwise適用されたクラス)
    """
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

        # subset/exclude 判定
        if mode == "subset" and (cid not in apply_classes):
            continue
        if mode == "exclude" and (cid in exclude_classes):
            continue
        if cid in exclude_classes:
            continue

        sim_c = float(v.get("thr_sim", thr_sim_g))
        m_c = float(v.get("thr_margin", thr_m_g))

        # clip（極端な値を抑える）
        if clip_sim > 0:
            sim_c = float(np.clip(sim_c, thr_sim_g - clip_sim, thr_sim_g + clip_sim))
        if clip_m > 0:
            m_c = float(np.clip(m_c, thr_m_g - clip_m, thr_m_g + clip_m))

        # apply mode
        if mode == "looser_sim_only":
            # classwise が “globalより甘い(sim閾値が低い)” 場合だけ採用（ただしclip後）
            thr_sim_table[cid] = min(thr_sim_g, sim_c)
            thr_m_table[cid] = thr_m_g  # marginはglobal固定（おすすめ）
            applied[cid] = 1

        elif mode == "looser_only":
            thr_sim_table[cid] = min(thr_sim_g, sim_c)
            if use_margin:
                thr_m_table[cid] = min(thr_m_g, m_c)
            else:
                thr_m_table[cid] = thr_m_g
            applied[cid] = 1

        else:
            # "all" / "subset" / "exclude" -> そのまま（ただしclip後）
            thr_sim_table[cid] = sim_c
            thr_m_table[cid] = m_c if use_margin else thr_m_g
            applied[cid] = 1

    # exclude は明示的に global に戻す
    for cid in exclude_classes:
        if 0 <= cid < num_classes:
            thr_sim_table[cid] = thr_sim_g
            thr_m_table[cid] = thr_m_g
            applied[cid] = 0

    return thr_sim_table, thr_m_table, applied


# ===================================
# main
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # ----------------------------
    # merge train cfg + infer cfg
    # ----------------------------
    infer_cfg = OmegaConf.to_container(cfg["300_infer"], resolve=True)
    train_exp = str(infer_cfg["exp"])
    train_cfg = load_train_yaml(infer_cfg["train_output_dir"], train_exp)
    # 100_train_arcface 配下に入ってる場合
    if isinstance(train_cfg, dict) and "100_train_arcface" in train_cfg:
        train_cfg = train_cfg["100_train_arcface"]

    merged = OmegaConf.merge(OmegaConf.create(train_cfg), OmegaConf.create(infer_cfg))
    config = dict_to_namespace(OmegaConf.to_container(merged, resolve=True))

    # ★debugでも train_exp を壊さない
    infer_name = str(getattr(config, "infer_name", "300_infer"))
    if bool(getattr(config, "debug", False)):
        infer_name = f"{infer_name}_debug"

    exp_dir = Path(config.train_output_dir) / train_exp
    outdir = exp_dir / "inference" / infer_name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "yaml").mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, outdir / "yaml" / "300_config.yaml")

    # ----------------------------
    # thresholds
    # ----------------------------
    combine_mode, global_thr, classwise = load_thresholds(exp_dir, config)
    thr_sim_table, thr_m_table, applied_table = build_threshold_tables(
        num_classes=int(config.num_classes),
        global_thr=global_thr,
        classwise=classwise,
        config=config,
    )

    # 保存（再現性）
    thr_df = pd.DataFrame({
        "class_id": np.arange(int(config.num_classes)),
        "thr_sim": thr_sim_table,
        "thr_margin": thr_m_table,
        "classwise_applied": applied_table.astype(int),
    })
    thr_df.to_csv(outdir / "threshold_table.csv", index=False)

    # ----------------------------
    # device
    # ----------------------------
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # csv paths
    # ----------------------------
    use_clean = bool(getattr(config, "use_clean_train_csv", True))
    train_csv_clean = exp_dir / "train_meta_pp_clean.csv"
    if use_clean and train_csv_clean.exists():
        train_csv = train_csv_clean
    else:
        # fallback: train_pp_csv が config にあればそれを使う / なければ pp_dir/pp_exp
        if hasattr(config, "train_pp_csv") and str(getattr(config, "train_pp_csv")):
            train_csv = Path(str(getattr(config, "train_pp_csv")))
        else:
            train_csv = Path(config.pp_dir) / config.pp_exp / "train_meta_pp.csv"

    test_csv = Path(config.pp_dir) / config.pp_exp / "test_meta_pp.csv"

    print(f"[csv] train_csv: {train_csv}")
    print(f"[csv] test_csv : {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # drop junk if exists（念のため）
    if "is_junk" in train_df.columns:
        before = len(train_df)
        train_df = train_df[train_df["is_junk"].fillna(0).astype(int) == 0].reset_index(drop=True)
        print(f"[train_df] drop is_junk: {before} -> {len(train_df)}")

    num_features = list(config.num_features) if bool(getattr(config, "use_tabular", True)) else []
    train_df = prepare_df(train_df, num_features)
    test_df = prepare_df(test_df, num_features)

    # ----------------------------
    # dataset / loader
    # ----------------------------
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
        train_ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    # ----------------------------
    # load module
    # ----------------------------
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
    module.setup("fit")  # EMA対策

    col2idx = {c: i for i, c in enumerate(num_features)}

    # ----------------------------
    # prototypes
    # ----------------------------
    print("computing prototypes...")
    prototypes = compute_prototypes(module, train_loader, device, num_classes=int(config.num_classes))
    np.save(outdir / "prototypes.npy", prototypes.numpy())

    # ----------------------------
    # embeddings
    # ----------------------------
    print("extracting test embeddings...")
    test_emb = extract_embeddings(
        module,
        test_loader,
        device,
        tta_hflip=bool(getattr(config, "tta_hflip", True)),
        col2idx=col2idx,
    )
    np.save(outdir / "test_emb.npy", test_emb.numpy())

    # ----------------------------
    # cosine similarity (N,C)
    # ----------------------------
    sims = test_emb.numpy() @ prototypes.numpy().T

    # top-1 / top-2
    top2_idx = np.argpartition(-sims, kth=1, axis=1)[:, :2]
    top2_val = np.take_along_axis(sims, top2_idx, axis=1)
    order = np.argsort(-top2_val, axis=1)

    top1 = top2_val[np.arange(len(sims)), order[:, 0]]
    top2 = top2_val[np.arange(len(sims)), order[:, 1]]
    pred = top2_idx[np.arange(len(sims)), order[:, 0]].astype(int)
    margin = (top1 - top2).astype(np.float32)

    # ----------------------------
    # unknown decision (improved)
    # ----------------------------
    thr_sim_used = thr_sim_table[pred]
    thr_m_used = thr_m_table[pred]

    unknown_by_sim = top1 < thr_sim_used

    # margin gate（sim が “低い領域” のときだけ margin を見る）
    # - margin_gate_sim があれば固定値
    # - なければ margin_gate_add が >0 なら (thr_sim_used + add)
    # - どちらも無ければゲート無し（従来通り）
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

    # ----------------------------
    # save
    # ----------------------------
    sub = pd.DataFrame({"label_id": pred_out})
    sub.to_csv(outdir / "submission.csv", index=False)

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
    dbg.to_csv(outdir / "test_pred_debug.csv", index=False)

    # summary（クラス別 unknown率）
    summary = (
        dbg.assign(pred_raw=dbg["pred_raw"].astype(int))
           .groupby("pred_raw")
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
    summary.to_csv(outdir / "pred_class_summary.csv", index=False)

    print("saved:", outdir)
    print("pred distribution:\n", pd.Series(pred_out).value_counts().sort_index())
    print("[unknown] rate:", float((pred_out == -1).mean()))
    print("[unknown parts] sim:", float(unknown_by_sim.mean()), "margin:", float(unknown_by_margin.mean()))


if __name__ == "__main__":
    main()