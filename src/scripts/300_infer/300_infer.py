from __future__ import annotations

from pathlib import Path
import json
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
    """
    学習expの保存済みyamlを読む。
    例: experiment/<exp>/yaml/config.yaml
    """
    p = Path(train_output_dir) / exp / "yaml" / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"train config yaml not found: {p}")
    return OmegaConf.to_container(OmegaConf.load(p), resolve=True)


def load_best_2d_txt(path: Path) -> dict[str, str]:
    """
    200の best_2d.txt をパースして dict で返す
    """
    if not path.exists():
        raise FileNotFoundError(f"best_2d.txt not found: {path}")
    d: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        d[k.strip()] = v.strip()
    return d


def prepare_df(df: pd.DataFrame, num_features: list[str]) -> pd.DataFrame:
    df = df.copy()

    # angle_id
    if "angle" in df.columns:
        df["angle_id"] = (df["angle"] == "top").astype(np.float32)
    else:
        df["angle_id"] = 0.0

    # n_players（なければ作る。あればそのまま使う）
    if "n_players" not in df.columns:
        if set(["quarter", "session", "frame", "angle"]).issubset(df.columns):
            df["n_players"] = (
                df.groupby(["quarter", "session", "frame"], sort=False)["angle"]
                  .transform("size")
                  .astype(np.int32)
            )
        else:
            df["n_players"] = 10
    df["n_players"] = df["n_players"].fillna(10).astype(int)

    # NaN埋め（num_featuresに含まれる列だけ）
    for c in num_features:
        if c not in df.columns:
            raise KeyError(f"num_feature not found: {c}")
        df[c] = df[c].fillna(0.0)

    # bbox_area の log1p は pp 側で済ませる想定（ここでは何もしない）
    return df


def flip_num_feats_batch(
    x: torch.Tensor,
    n_players: torch.Tensor,
    col2idx: dict[str, int]
) -> torch.Tensor:
    """
    x: (B,F) float
    n_players: (B,) int
    """
    x = x.clone()
    if "fx" in col2idx:
        x[:, col2idx["fx"]] = -x[:, col2idx["fx"]]
    if "rank_x" in col2idx:
        rx = x[:, col2idx["rank_x"]]
        x[:, col2idx["rank_x"]] = (n_players.float() - 1.0) - rx
    return x


@torch.no_grad()
def compute_prototypes(
    module: PlayerLightningModule,
    loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> torch.Tensor:
    """
    各クラスの平均埋め込み（L2正規化済み）を作る
    """
    module.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["num"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True).long()

        if module.use_ema and module.model_ema is not None:
            emb = module.model_ema.module.get_embedding(img, tab)
        else:
            emb = module.model.get_embedding(img, tab)

        emb = F.normalize(emb.float(), p=2, dim=1)  # float32で安定

        if sums is None:
            sums = torch.zeros(num_classes, emb.size(1), device=device, dtype=torch.float32)

        # sums[c] += emb[i] where y[i]==c
        sums.index_add_(0, y, emb)
        counts += torch.bincount(y, minlength=num_classes).float()

    prot = sums / counts.clamp(min=1.0).unsqueeze(1)
    prot = F.normalize(prot, p=2, dim=1)
    return prot.detach().cpu()


@torch.no_grad()
def extract_embeddings(
    module: PlayerLightningModule,
    loader: DataLoader,
    device: torch.device,
    tta_hflip: bool,
    col2idx: dict[str, int]
) -> torch.Tensor:
    module.eval()
    embs: list[torch.Tensor] = []

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["num"].to(device, non_blocking=True)

        n_players = batch.get("n_players", None)
        if n_players is None:
            n_players = torch.full((img.size(0),), 10, device=device, dtype=torch.long)
        else:
            n_players = n_players.to(device, non_blocking=True).long()

        # orig
        if module.use_ema and module.model_ema is not None:
            e1 = module.model_ema.module.get_embedding(img, tab)
        else:
            e1 = module.model.get_embedding(img, tab)
        e1 = F.normalize(e1.float(), p=2, dim=1)

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
        e2 = F.normalize(e2.float(), p=2, dim=1)

        e = F.normalize((e1 + e2) * 0.5, p=2, dim=1)
        embs.append(e.cpu())

    return torch.cat(embs, dim=0)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


# ===================================
# main
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # -----------------------
    # merge train config + infer config
    # -----------------------
    infer_cfg = OmegaConf.to_container(cfg["300_infer"], resolve=True)
    train_cfg = load_train_yaml(infer_cfg["train_output_dir"], infer_cfg["exp"])

    # 100_train_arcface の入れ子にしている場合に対応
    if "100_train_arcface" in train_cfg:
        train_cfg = train_cfg["100_train_arcface"]

    merged = OmegaConf.merge(OmegaConf.create(train_cfg), OmegaConf.create(infer_cfg))
    config = dict_to_namespace(OmegaConf.to_container(merged, resolve=True))

    # -----------------------
    # paths (exp配下に全部集約)
    # -----------------------
    exp_dir = Path(config.train_output_dir) / str(config.exp)
    outdir = exp_dir / ("inference_debug" if bool(getattr(config, "debug", False)) else "inference")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "yaml").mkdir(parents=True, exist_ok=True)

    save_config_yaml(config, outdir / "yaml" / "300_config.yaml")

    # device / seed（推論は基本deterministicだが一応）
    if hasattr(config, "seed"):
        np.random.seed(int(config.seed))
        torch.manual_seed(int(config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.seed))

    device = torch.device(str(config.device) if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # preprocess csv paths（infer側の pp_dir/pp_exp を優先）
    pp_dir = Path(config.pp_dir)
    pp_exp = str(config.pp_exp)
    train_pp_csv = pp_dir / pp_exp / "train_meta_pp.csv"
    test_pp_csv = pp_dir / pp_exp / "test_meta_pp.csv"
    if not train_pp_csv.exists():
        raise FileNotFoundError(f"train_meta_pp.csv not found: {train_pp_csv}")
    if not test_pp_csv.exists():
        raise FileNotFoundError(f"test_meta_pp.csv not found: {test_pp_csv}")

    # ckpt path（expから自動）
    ckpt_path = Path(config.train_output_dir) / str(config.exp) / "model" / "best.ckpt"
    if hasattr(config, "ckpt_path") and str(getattr(config, "ckpt_path")):
        # 明示指定があるならそちら優先
        ckpt_path = Path(str(getattr(config, "ckpt_path")))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # -----------------------
    # (optional) load thresholds from 200 result
    # -----------------------
    # 200で出した best_2d.txt を読んで unknown_threshold/margin_threshold を自動セット
    if bool(getattr(config, "load_best_threshold", False)):
        best_kind = str(getattr(config, "best_threshold_kind", "median_nonzero"))  # "median" or "median_nonzero"
        best_file = str(getattr(config, "best_threshold_file", "best_2d.txt"))
        best_path = exp_dir / "threshold" / best_file

        d = load_best_2d_txt(best_path)
        if best_kind == "median":
            thr = float(d["thr_median"])
            mthr = float(d["mthr_median"])
        elif best_kind == "median_nonzero":
            thr = float(d["thr_median_nonzero"])
            mthr = float(d["mthr_median_nonzero"])
        else:
            raise ValueError(f"unknown best_threshold_kind: {best_kind}")

        config.unknown_threshold = thr
        config.margin_threshold = mthr

        # combine_mode も best_2d に書かれてる場合、yaml未指定ならそれを使う（保険）
        if not hasattr(config, "combine_mode") and "combine_mode" in d:
            config.combine_mode = str(d["combine_mode"])

        print(f"[threshold] loaded from {best_path} ({best_kind}): thr={thr}, mthr={mthr}")

    # -----------------------
    # load dataframes
    # -----------------------
    train_df = pd.read_csv(train_pp_csv)
    test_df = pd.read_csv(test_pp_csv)

    num_features = list(config.num_features) if bool(getattr(config, "use_tabular", False)) else []
    train_df = prepare_df(train_df, num_features)
    test_df = prepare_df(test_df, num_features)

    # Dataset / Loader (no augmentation)
    tf = get_val_transform(int(config.img_size))

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

    # -----------------------
    # load model
    # -----------------------
    module = PlayerLightningModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model_name=str(config.model_name),
        num_classes=int(config.num_classes),
        num_tabular_features=len(num_features) if bool(getattr(config, "use_tabular", False)) else 0,
        embedding_dim=int(config.embedding_dim),
        pretrained=False,
        arcface_s=float(config.arcface_s),
        arcface_m=float(config.arcface_m),
        lr=1e-3,
        weight_decay=0.0,
        epochs=1,
        use_ema=bool(getattr(config, "use_ema", False)),
        ema_decay=float(getattr(config, "ema_decay", 0.999)),
    )
    module = module.to(device)
    module.eval()

    # ensure EMA initialized (Lightning setup isn't called here)
    module.setup("fit")

    col2idx = {c: i for i, c in enumerate(num_features)}

    # -----------------------
    # cache（メタ一致なら再利用）
    # -----------------------
    reuse_cache = bool(getattr(config, "reuse_cache", True))

    proto_path = outdir / "prototypes.npy"
    proto_meta_path = outdir / "prototypes_meta.json"
    proto_meta = {
        "ckpt_path": str(ckpt_path),
        "train_pp_csv": str(train_pp_csv),
        "img_size": int(config.img_size),
        "use_tabular": bool(getattr(config, "use_tabular", False)),
        "num_features": num_features,
    }

    emb_path = outdir / "test_emb.npy"
    emb_meta_path = outdir / "test_emb_meta.json"
    emb_meta = {
        "ckpt_path": str(ckpt_path),
        "test_pp_csv": str(test_pp_csv),
        "img_size": int(config.img_size),
        "use_tabular": bool(getattr(config, "use_tabular", False)),
        "num_features": num_features,
        "tta_hflip": bool(getattr(config, "tta_hflip", False)),
    }

    # -----------------------
    # prototypes
    # -----------------------
    if reuse_cache and proto_path.exists() and _load_json(proto_meta_path) == proto_meta:
        print("loading cached prototypes:", proto_path)
        prototypes = torch.from_numpy(np.load(proto_path)).float()
    else:
        print("computing prototypes...")
        prototypes = compute_prototypes(module, train_loader, device, num_classes=int(config.num_classes)).float()
        np.save(proto_path, prototypes.numpy())
        _save_json(proto_meta_path, proto_meta)

    # -----------------------
    # test embeddings
    # -----------------------
    if reuse_cache and emb_path.exists() and _load_json(emb_meta_path) == emb_meta:
        print("loading cached test embeddings:", emb_path)
        test_emb = torch.from_numpy(np.load(emb_path)).float()
    else:
        print("extracting test embeddings...")
        test_emb = extract_embeddings(
            module,
            test_loader,
            device,
            tta_hflip=bool(getattr(config, "tta_hflip", False)),
            col2idx=col2idx,
        ).float()
        np.save(emb_path, test_emb.numpy())
        _save_json(emb_meta_path, emb_meta)

    # -----------------------
    # similarity / prediction
    # -----------------------
    print("computing cosine similarity & topk...")
    sims = test_emb @ prototypes.T  # (N, C)

    top2_val, top2_idx = torch.topk(sims, k=2, dim=1, largest=True)
    top1 = top2_val[:, 0]
    top2 = top2_val[:, 1]
    pred = top2_idx[:, 0]
    margin = top1 - top2

    thr1 = float(getattr(config, "unknown_threshold", 0.0))
    thr2 = float(getattr(config, "margin_threshold", 0.0))
    combine_mode = str(getattr(config, "combine_mode", "or"))  # "or" or "and"

    cond_sim = top1 < thr1
    cond_m = margin < thr2

    if combine_mode == "or":
        unknown_cond = cond_sim | cond_m
    elif combine_mode == "and":
        unknown_cond = cond_sim & cond_m
    else:
        raise ValueError(f"unknown combine_mode: {combine_mode}")

    pred_out = pred.clone()
    pred_out[unknown_cond] = -1

    pred_out_np = pred_out.cpu().numpy().astype(int)

    # submission
    sub = pd.DataFrame({"label_id": pred_out_np})
    sub.to_csv(outdir / "submission.csv", index=False)

    # debug
    dbg = pd.DataFrame({
        "max_sim": top1.cpu().numpy(),
        "second_sim": top2.cpu().numpy(),
        "margin": margin.cpu().numpy(),
        "pred_raw": pred.cpu().numpy(),
        "pred": pred_out_np,
    })
    dbg.to_csv(outdir / "test_pred_debug.csv", index=False)

    # thresholds used
    with open(outdir / "used_threshold.txt", "w") as f:
        f.write(f"unknown_threshold={thr1}\n")
        f.write(f"margin_threshold={thr2}\n")
        f.write(f"combine_mode={combine_mode}\n")
        f.write(f"reuse_cache={reuse_cache}\n")
        f.write(f"ckpt_path={ckpt_path}\n")
        f.write(f"pp_exp={pp_exp}\n")

    print("saved:", outdir)
    print("pred distribution:\n", pd.Series(pred_out_np).value_counts().sort_index())


if __name__ == "__main__":
    main()