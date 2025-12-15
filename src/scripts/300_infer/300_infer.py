import os
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
def load_train_yaml(train_output_dir: str, train_exp: str) -> dict:
    """
    学習expの保存済みyamlを読む。
    例: experiment/<train_exp>/yaml/config.yaml
    """
    p = Path(train_output_dir) / train_exp / "yaml" / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"train config yaml not found: {p}")
    return OmegaConf.to_container(OmegaConf.load(p), resolve=True)

def prepare_df(df: pd.DataFrame, num_features: list[str]) -> pd.DataFrame:
    df = df.copy()

    # angle_id
    df["angle_id"] = (df["angle"] == "top").astype(np.float32)

    # n_players（なければ作る。あればそのまま使う）
    if "n_players" not in df.columns:
        df["n_players"] = (
            df.groupby(["quarter", "session", "frame"], sort=False)["angle"]
              .transform("size")
              .astype(np.int32)
        )
    df["n_players"] = df["n_players"].fillna(10).astype(int)

    # NaN埋め（num_featuresに含まれる列だけ）
    for c in num_features:
        if c not in df.columns:
            raise KeyError(f"num_feature not found: {c}")
        df[c] = df[c].fillna(0.0)

    # bbox_area が未logなら log1p（既にlogなら二重変換なので注意）
    # ここは「pp側でlog済み」ならコメントアウト推奨
    if "bbox_area" in df.columns:
        # すでにlog1p済みなら二重になるので、必要ならフラグで制御するのが理想
        df["bbox_area"] = np.log1p(np.clip(df["bbox_area"].astype(np.float32), 0, None))

    return df


def flip_num_feats_batch(x: torch.Tensor, n_players: torch.Tensor, col2idx: dict[str,int]) -> torch.Tensor:
    """
    x: (B,F) float
    n_players: (B,) int
    """
    x = x.clone()
    if "fx" in col2idx:
        x[:, col2idx["fx"]] = -x[:, col2idx["fx"]]
    if "rank_x" in col2idx:
        # rank_x -> (n-1-rank_x)
        rx = x[:, col2idx["rank_x"]]
        x[:, col2idx["rank_x"]] = (n_players.float() - 1.0) - rx
    return x

@torch.no_grad()
def compute_prototypes(module: PlayerLightningModule, loader: DataLoader, device, num_classes: int) -> torch.Tensor:
    module.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)  # ★deviceへ & float

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

        # orig
        if module.use_ema and module.model_ema is not None:
            e1 = module.model_ema.module.get_embedding(img, tab)
        else:
            e1 = module.model.get_embedding(img, tab)
        e1 = F.normalize(e1, p=2, dim=1)

        if not tta_hflip:
            embs.append(e1.cpu())
            continue

        # flip
        img_f = torch.flip(img, dims=[3])  # (B,C,H,W)
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
# TODO: config_pathをこのスクリプトからの相対パスにする
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """description
    Args:
        cfg (DictConf): config
    """
    # set config
    config_dict = OmegaConf.to_container(cfg["300_infer"], resolve=True)
    config = dict_to_namespace(config_dict)
    infer_cfg = OmegaConf.to_container(cfg["300_infer"], resolve=True)
    train_cfg = load_train_yaml(infer_cfg["train_output_dir"], infer_cfg["train_exp"])
    if "100_train_arcface" in train_cfg:
        train_cfg = train_cfg["100_train_arcface"]
    merged = OmegaConf.merge(OmegaConf.create(train_cfg), OmegaConf.create(infer_cfg))
    config = dict_to_namespace(OmegaConf.to_container(merged, resolve=True))

    # when debug
    if config.debug:
        config.exp = "300_infer_debug" # TODO: ファイルの連番を入れる
    outdir = Path(config.output_dir) / config.exp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "yaml").mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, outdir / "yaml" / "config.yaml")
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    # preprocess csv paths（infer側の pp_dir/pp_exp を優先）
    pp_dir = Path(config.pp_dir)
    pp_exp = config.pp_exp
    config.train_pp_csv = str(pp_dir / pp_exp / "train_meta_pp.csv")
    config.test_pp_csv  = str(pp_dir / pp_exp / "test_meta_pp.csv")
    # ckpt path / output_dir（train_expから自動）
    config.ckpt_path = str(Path(config.train_output_dir) / config.train_exp / "model" / "best.ckpt")
    config.output_dir = str(Path(config.train_output_dir) / config.train_exp / "inference")
    # load csv
    train_df = pd.read_csv(config.train_pp_csv)
    test_df = pd.read_csv(config.test_pp_csv)

    num_features = list(config.num_features) if config.use_tabular else []
    train_df = prepare_df(train_df, num_features)
    test_df = prepare_df(test_df, num_features)

    # Dataset / Loader (no augmentation)
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
    module = PlayerLightningModule.load_from_checkpoint(
        checkpoint_path=str(config.ckpt_path),
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

    # ensure EMA initialized (Lightning setup isn't called here)
    module.setup("fit")

    col2idx = {c:i for i,c in enumerate(num_features)}

    # prototypes
    print("computing prototypes...")
    prototypes = compute_prototypes(module, train_loader, device, num_classes=config.num_classes)
    np.save(outdir / "prototypes.npy", prototypes.numpy())

    # embeddings
    print("extracting test embeddings...")
    test_emb = extract_embeddings(module, test_loader, device, tta_hflip=bool(config.tta_hflip), col2idx=col2idx)
    np.save(outdir / "test_emb.npy", test_emb.numpy())

    # cosine similarity
    sims = test_emb.numpy() @ prototypes.numpy().T  # (N,C)
    max_sim = sims.max(axis=1)
    pred = sims.argmax(axis=1)

    # unknown threshold
    thr = float(config.unknown_threshold)
    pred_out = np.where(max_sim < thr, -1, pred).astype(int)

    # submission (test_df row order is submission order)
    sub = pd.DataFrame({"label_id": pred_out})
    sub.to_csv(outdir / "submission.csv", index=False)

    # debug info
    dbg = pd.DataFrame({
        "max_sim": max_sim,
        "pred_raw": pred,
        "pred": pred_out,
    })
    dbg.to_csv(outdir / "test_pred_debug.csv", index=False)

    print("saved:", outdir)
    print("pred distribution:\n", pd.Series(pred_out).value_counts().sort_index())


if __name__ == "__main__":
    main()
