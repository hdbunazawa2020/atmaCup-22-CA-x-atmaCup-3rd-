from pathlib import Path
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


def load_best_threshold(exp_dir: Path, filename: str, thr_key: str, mthr_key: str) -> tuple[float, float]:
    p = exp_dir / "threshold" / filename
    if not p.exists():
        raise FileNotFoundError(f"threshold file not found: {p}")

    kv = {}
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    if thr_key not in kv or mthr_key not in kv:
        raise KeyError(f"keys not found in {p}: need {thr_key},{mthr_key}. got {list(kv.keys())}")

    return float(kv[thr_key]), float(kv[mthr_key])


def prepare_df(df: pd.DataFrame, num_features: list[str]) -> pd.DataFrame:
    df = df.copy()

    # angle_id（無ければ作る）
    if "angle_id" not in df.columns:
        df["angle_id"] = (df["angle"] == "top").astype(np.float32)

    # n_players（無ければ作る。※angle込みで数えるのが重要）
    if "n_players" not in df.columns:
        df["n_players"] = (
            df.groupby(["quarter", "session", "frame", "angle"], sort=False)["angle"]
              .transform("size")
              .astype(np.int32)
        )
    df["n_players"] = df["n_players"].fillna(10).astype(int)

    # NaN埋め（num_featuresのみ）
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
    if "rank_x_norm" in col2idx:
        rxn = x[:, col2idx["rank_x_norm"]]
        denom = (n_players.float() - 1.0)
        x[:, col2idx["rank_x_norm"]] = torch.where(denom > 0, 1.0 - rxn, torch.zeros_like(rxn))
    return x


@torch.no_grad()
def compute_prototypes(module: PlayerLightningModule, loader: DataLoader, device, num_classes: int) -> torch.Tensor:
    module.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["num"].to(device, non_blocking=True)
        y   = batch["label"].to(device, non_blocking=True).long()

        if module.use_ema and module.model_ema is not None:
            emb = module.model_ema.module.get_embedding(img, tab)
        else:
            emb = module.model.get_embedding(img, tab)

        emb = F.normalize(emb, p=2, dim=1)

        if sums is None:
            sums = torch.zeros(num_classes, emb.size(1), device=device, dtype=emb.dtype)

        sums.index_add_(0, y, emb)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))

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
            if "n_players" in col2idx:
                n_players = tab[:, col2idx["n_players"]].round().long()
            else:
                n_players = torch.full((img.size(0),), 10, device=device, dtype=torch.long)
        else:
            n_players = n_players.to(device, non_blocking=True).long()

        # orig
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


# ===================================
# main
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    infer_cfg = OmegaConf.to_container(cfg["300_infer"], resolve=True)

    # train cfg merge（モデル設定はtrain側、推論設定はinfer側を優先）
    train_cfg = load_train_yaml(infer_cfg["train_output_dir"], infer_cfg["exp"])
    if "100_train_arcface" in train_cfg:
        train_cfg = train_cfg["100_train_arcface"]
    merged = OmegaConf.merge(OmegaConf.create(train_cfg), OmegaConf.create(infer_cfg))
    config = dict_to_namespace(OmegaConf.to_container(merged, resolve=True))

    exp_dir = Path(config.train_output_dir) / config.exp
    infer_exp = str(getattr(config, "infer_exp", "300_infer"))
    outdir = exp_dir / "inference" / infer_exp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "yaml").mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, outdir / "yaml" / "300_config.yaml")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # ---- thresholds (optional auto-load from 200 output) ----
    if bool(getattr(config, "load_threshold_from_200", False)):
        thr_key = str(getattr(config, "thr_key", "thr_median_nonzero"))
        mthr_key = str(getattr(config, "mthr_key", "mthr_median_nonzero"))
        fname = str(getattr(config, "threshold_file", "best_2d.txt"))
        thr1, thr2 = load_best_threshold(exp_dir, fname, thr_key, mthr_key)
        print(f"[threshold] loaded from {exp_dir/'threshold'/fname}: {thr_key}={thr1}, {mthr_key}={thr2}")
    else:
        thr1 = float(config.unknown_threshold)
        thr2 = float(getattr(config, "margin_threshold", 0.0))

    combine_mode = str(getattr(config, "combine_mode", "and"))
    print("[threshold] combine_mode:", combine_mode, "thr_sim:", thr1, "thr_margin:", thr2)

    # ---- preprocess csv paths ----
    pp_dir = Path(config.pp_dir)
    pp_exp = config.pp_exp

    train_csv_default = pp_dir / pp_exp / "train_meta_pp.csv"
    test_csv = pp_dir / pp_exp / "test_meta_pp.csv"

    # train prototypes 用: is_junk を落とした clean があれば優先
    use_clean = bool(getattr(config, "use_clean_train_csv", True))
    clean_csv = exp_dir / "train_meta_pp_clean.csv"
    train_csv = clean_csv if (use_clean and clean_csv.exists()) else train_csv_default

    print("[csv] train_csv:", train_csv)
    print("[csv] test_csv :", test_csv)

    # ckpt path
    ckpt_path = exp_dir / "model" / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best.ckpt not found: {ckpt_path}")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # 念のため: is_junk が残ってたら落とす
    if "is_junk" in train_df.columns:
        before = len(train_df)
        train_df = train_df[train_df["is_junk"].fillna(0).astype(int) == 0].copy()
        print(f"[train_df] drop is_junk: {before} -> {len(train_df)}")

    num_features = list(config.num_features) if getattr(config, "use_tabular", True) else []
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
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    module = PlayerLightningModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model_name=config.model_name,
        num_classes=config.num_classes,
        num_tabular_features=len(num_features) if getattr(config, "use_tabular", True) else 0,
        embedding_dim=config.embedding_dim,
        pretrained=False,
        arcface_s=config.arcface_s,
        arcface_m=config.arcface_m,
        lr=1e-3,
        weight_decay=0.0,
        epochs=1,
        use_ema=config.use_ema,
        ema_decay=config.ema_decay,
    ).to(device)
    module.eval()
    module.setup("fit")  # EMA init

    col2idx = {c: i for i, c in enumerate(num_features)}

    print("computing prototypes...")
    prototypes = compute_prototypes(module, train_loader, device, num_classes=config.num_classes)
    np.save(outdir / "prototypes.npy", prototypes.numpy())

    print("extracting test embeddings...")
    test_emb = extract_embeddings(
        module,
        test_loader,
        device,
        tta_hflip=bool(config.tta_hflip),
        col2idx=col2idx,
    )
    np.save(outdir / "test_emb.npy", test_emb.numpy())

    sims = test_emb.numpy() @ prototypes.numpy().T  # (N, C)

    top2_idx = np.argpartition(-sims, kth=1, axis=1)[:, :2]
    top2_val = np.take_along_axis(sims, top2_idx, axis=1)
    order = np.argsort(-top2_val, axis=1)

    top1 = top2_val[np.arange(len(sims)), order[:, 0]]
    top2 = top2_val[np.arange(len(sims)), order[:, 1]]
    pred = top2_idx[np.arange(len(sims)), order[:, 0]]
    margin = top1 - top2

    cond_sim = (top1 < thr1)
    cond_m   = (margin < thr2)
    if combine_mode == "and":
        unknown_cond = cond_sim & cond_m
    elif combine_mode == "or":
        unknown_cond = cond_sim | cond_m
    else:
        raise ValueError(f"unknown combine_mode: {combine_mode}")

    pred_out = np.where(unknown_cond, -1, pred).astype(int)

    sub = pd.DataFrame({"label_id": pred_out})
    sub.to_csv(outdir / "submission.csv", index=False)

    dbg = pd.DataFrame({
        "max_sim": top1,
        "second_sim": top2,
        "margin": margin,
        "pred_raw": pred,
        "pred": pred_out,
    })
    dbg.to_csv(outdir / "test_pred_debug.csv", index=False)

    print("saved:", outdir)
    print("pred distribution:\n", pd.Series(pred_out).value_counts().sort_index())


if __name__ == "__main__":
    main()