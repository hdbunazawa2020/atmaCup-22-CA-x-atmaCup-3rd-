import os, re, warnings, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2

import hydra
from omegaconf import DictConfig, OmegaConf

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# original
import sys
sys.path.append(r"..")
from utils.data import sep, show_df, save_config_yaml, dict_to_namespace

warnings.filterwarnings("ignore")

# ===================================
# image path utils
# ===================================
def get_image_path(row: pd.Series, image_dir: Path) -> Path:
    """quarter__angle__session__frame.jpg を返す"""
    fname = f"{row['quarter']}__{row['angle']}__{int(row['session']):02d}__{int(row['frame']):02d}.jpg"
    return image_dir / fname


def process_single_crop(args: tuple) -> tuple[int, bool, str]:
    """
    1行分のbboxをクロップして保存。
    Returns: (idx, success, saved_path_str)
    """
    idx, row, image_dir, output_dir, padding_ratio, prefix = args
    try:
        img_path = get_image_path(row, image_dir)
        img = cv2.imread(str(img_path))
        if img is None:
            return idx, False, ""

        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        img_h, img_w = img.shape[:2]

        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return idx, False, ""

        # 衝突しないファイル名
        out_name = f"{prefix}_{idx}.jpg"
        output_path = output_dir / out_name
        cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return idx, True, str(output_path)
    except Exception as e:
        print(f"[crop error] idx={idx} err={e}")
        return idx, False, ""


def preprocess_crops(
    csv_path: Path,
    image_dir: Path,
    output_dir: Path,
    prefix: str,
    padding_ratio: float = 0.10,
    num_workers: int | None = None,
) -> pd.DataFrame:
    """
    csvを読み、全bboxをクロップして output_dir に保存。
    返り値: 元csv + crop_path列（失敗はNaN）
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["session"] = df["session"].astype(int)
    df["frame"] = df["frame"].astype(int)

    print(f"[{prefix}] {num_workers} workers, samples={len(df)} -> {output_dir}")

    args_list = [(idx, row, image_dir, output_dir, padding_ratio, prefix) for idx, row in df.iterrows()]

    crop_paths = {}
    failed = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_crop, args): args[0] for args in args_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"cropping {prefix}"):
            idx, ok, out_path = fut.result()
            if ok:
                crop_paths[idx] = out_path
            else:
                failed.append(idx)

    df["crop_path"] = df.index.map(crop_paths).astype("object")
    miss = df["crop_path"].isna().mean()
    print(f"[{prefix}] crop_path missing rate: {miss:.6f}")
    if failed:
        print(f"[{prefix}] failed idx sample: {failed[:10]}")

    return df


# ===================================
# image presence meta
# ===================================
def build_image_presence_table(img_dir: Path) -> pd.DataFrame:
    """
    画像ファイル名: {quarter}__{angle}__{session}__{frame}.jpg
    戻り値:
      quarter, session, frame, is_top, is_side, top_path, side_path
    """
    rows = []
    bad = []
    exts = {".jpg", ".jpeg"}

    files = [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    for p in files:
        parts = p.stem.split("__")
        if len(parts) != 4:
            bad.append(p.name)
            continue
        quarter, angle, session_str, frame_str = parts
        if angle not in ("top", "side"):
            bad.append(p.name)
            continue
        try:
            session = int(session_str)
            frame = int(frame_str)
        except ValueError:
            bad.append(p.name)
            continue
        rows.append({"quarter": quarter, "angle": angle, "session": session, "frame": frame, "path": str(p)})

    df_files = pd.DataFrame(rows)
    if len(df_files) == 0:
        raise RuntimeError(f"No valid images under: {img_dir}")

    df_pivot = (
        df_files.pivot_table(
            index=["quarter", "session", "frame"],
            columns="angle",
            values="path",
            aggfunc="first",
        )
        .rename(columns={"top": "top_path", "side": "side_path"})
        .reset_index()
    )

    if "top_path" not in df_pivot.columns:
        df_pivot["top_path"] = pd.NA
    if "side_path" not in df_pivot.columns:
        df_pivot["side_path"] = pd.NA

    df_pivot["is_top"] = df_pivot["top_path"].notna().astype("int8")
    df_pivot["is_side"] = df_pivot["side_path"].notna().astype("int8")

    df_pivot = df_pivot[["quarter", "session", "frame", "is_top", "is_side", "top_path", "side_path"]]
    df_pivot = df_pivot.sort_values(["quarter", "session", "frame"]).reset_index(drop=True)

    if bad:
        print(f"[WARN] skipped {len(bad)} files, example={bad[:5]}")

    return df_pivot


# ===================================
# formation features
# ===================================
def get_img_size_from_path(path: str):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"image not found: {path}")
    h, w = img.shape[:2]
    return w, h

def get_angle_img_sizes(df, angle_col="angle", path_col="image_path"):
    sizes = {}
    for ang, g in df.groupby(angle_col):
        p = g[path_col].dropna().iloc[0]
        sizes[ang] = get_img_size_from_path(p)
    return sizes

def bbox_anchor_xy(df, img_w, img_h):
    x = df["x"].to_numpy(np.float32)
    y = df["y"].to_numpy(np.float32)
    w = df["w"].to_numpy(np.float32)
    h = df["h"].to_numpy(np.float32)

    cx = x + 0.5 * w
    cy = np.where(df["angle"].to_numpy() == "top", y + 0.5*h, y + h)  # top:center, side:bottom

    nx = cx / float(img_w)
    ny = cy / float(img_h)
    return np.stack([nx, ny], axis=1).astype(np.float32)

def normalize_points_rigid(P, eps=1e-6):
    c = P.mean(axis=0, keepdims=True)
    X = P - c
    s = np.sqrt((X**2).sum(axis=1).mean()) + eps
    X = X / s

    C = (X.T @ X) / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    v1 = eigvecs[:, 1]
    v2 = eigvecs[:, 0]
    R = np.stack([v1, v2], axis=1)
    Q = X @ R

    if Q[:, 0].max() < -Q[:, 0].min():
        Q *= -1.0

    return Q.astype(np.float32), float(s)

def add_formation_features(df: pd.DataFrame, img_sizes: dict, key_cols=("quarter","session","frame")):
    df = df.copy()
    df["__row_id"] = np.arange(len(df), dtype=np.int64)

    out_chunks = []
    for _, g in df.groupby(list(key_cols), sort=False):
        g = g.copy()
        g["__pos"] = np.arange(len(g), dtype=np.int32)

        P = np.zeros((len(g), 2), dtype=np.float32)

        for ang, gg in g.groupby("angle", sort=False):
            img_w, img_h = img_sizes[ang]
            P_ang = bbox_anchor_xy(gg, img_w, img_h)
            P[gg["__pos"].to_numpy()] = P_ang

        Qn, shape_scale = normalize_points_rigid(P)

        center = P.mean(axis=0)
        dist_center = np.sqrt(((P - center) ** 2).sum(axis=1))

        rank_x = pd.Series(P[:, 0]).rank(method="first").to_numpy(np.int32) - 1
        rank_y = pd.Series(P[:, 1]).rank(method="first").to_numpy(np.int32) - 1

        if len(P) >= 2:
            D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)).astype(np.float32)
            np.fill_diagonal(D, np.nan)
            nn_dist = np.nanmin(D, axis=1)
            mean_dist = np.nanmean(D, axis=1)
        else:
            nn_dist = np.full((len(P),), np.nan, dtype=np.float32)
            mean_dist = np.full((len(P),), np.nan, dtype=np.float32)

        gg_out = g.drop(columns="__pos").copy()
        gg_out["fx"] = Qn[:, 0]
        gg_out["fy"] = Qn[:, 1]
        gg_out["dist_center"] = dist_center.astype(np.float32)
        gg_out["rank_x"] = rank_x.astype(np.int16)
        gg_out["rank_y"] = rank_y.astype(np.int16)
        gg_out["nn_dist"] = nn_dist.astype(np.float32)
        gg_out["mean_dist"] = mean_dist.astype(np.float32)
        gg_out["shape_scale"] = np.float32(shape_scale)
        gg_out["bbox_area"] = (gg_out["w"].to_numpy(np.float32) * gg_out["h"].to_numpy(np.float32))
        gg_out["bbox_aspect"] = (gg_out["w"].to_numpy(np.float32) / (gg_out["h"].to_numpy(np.float32) + 1e-6))

        out_chunks.append(gg_out)

    out = pd.concat(out_chunks, axis=0)
    out = out.sort_values("__row_id").drop(columns="__row_id").reset_index(drop=True)
    return out


# ===================================
# main
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg["000_preprocess"], resolve=True)
    config = dict_to_namespace(config_dict)

    if config.debug:
        config.exp = "000_pp_debug"
    savedir = Path(config.output_dir) / config.exp
    (savedir / "yaml").mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, savedir / "yaml" / "config.yaml")

    DATA_DIR  = Path(config.data_dir)
    IMAGE_DIR = Path(config.image_dir)

    # crops dirs
    CROP_ROOT = Path(config.crop_dir).parent  # ../data/processed/crops
    CROP_TRAIN = CROP_ROOT / "train"
    CROP_TEST = CROP_ROOT / "test"
    CROP_TEST_TOP = CROP_ROOT / "test_top"

    # ======================
    # 1) Build image meta (top/side presence)
    # ======================
    img_meta = build_image_presence_table(IMAGE_DIR)
    print(img_meta.groupby(["is_top","is_side"]).size().rename("n_frames"))
    sep("img_meta"); show_df(img_meta, 5, True)

    # ======================
    # 2) Read raw meta tables
    # ======================
    train_raw = pd.read_csv(DATA_DIR / "train_meta.csv")
    test_raw = pd.read_csv(DATA_DIR / "test_meta.csv")
    test_top_raw = pd.read_csv(DATA_DIR / "test_top_meta.csv")

    # dtype safety
    for df in (train_raw, test_raw, test_top_raw):
        df["session"] = df["session"].astype(int)
        df["frame"] = df["frame"].astype(int)

    # merge image paths (for img size lookup + optional debug)
    def attach_image_paths(df):
        df = df.merge(img_meta, on=["quarter","session","frame"], how="left", validate="m:1")
        df["image_path"] = np.where(df["angle"]=="top", df["top_path"], df["side_path"])
        df["paired_path"] = np.where(df["angle"]=="top", df["side_path"], df["top_path"])
        return df

    train = attach_image_paths(train_raw)
    test  = attach_image_paths(test_raw)
    test_top = attach_image_paths(test_top_raw)

    print("train image_path missing rate:", train["image_path"].isna().mean())
    print("test  image_path missing rate:",  test["image_path"].isna().mean())
    print("test_top image_path missing rate:", test_top["image_path"].isna().mean())

    # ======================
    # 3) Crop train/test/test_top (collision-free names) and add crop_path
    # ======================
    # ここで「csv -> crop_path付きdf」を作る
    if config.use_crops:
        # クロップは raw csv を使う（列が少なく高速）
        train_crop_df = preprocess_crops(
            csv_path=DATA_DIR / "train_meta.csv",
            image_dir=IMAGE_DIR,
            output_dir=CROP_TRAIN,
            prefix="train",
            padding_ratio=float(config.padding_ratio),
            num_workers=None,
        )
        test_crop_df = preprocess_crops(
            csv_path=DATA_DIR / "test_meta.csv",
            image_dir=IMAGE_DIR,
            output_dir=CROP_TEST,
            prefix="test",
            padding_ratio=float(config.padding_ratio),
            num_workers=None,
        )
        test_top_crop_df = preprocess_crops(
            csv_path=DATA_DIR / "test_top_meta.csv",
            image_dir=IMAGE_DIR,
            output_dir=CROP_TEST_TOP,
            prefix="testtop",
            padding_ratio=float(config.padding_ratio),
            num_workers=None,
        )

        # 元のtrain/test/test_topに crop_path を付与（行順一致で join）
        train["crop_path"] = train_crop_df["crop_path"].values
        test["crop_path"] = test_crop_df["crop_path"].values
        test_top["crop_path"] = test_top_crop_df["crop_path"].values
    else:
        train["crop_path"] = train["image_path"]  # fallback（非推奨）
        test["crop_path"] = test["image_path"]
        test_top["crop_path"] = test_top["image_path"]

    # ======================
    # 4) Formation features
    # ======================
    union_df = pd.concat(
        [
            train[["angle","image_path"]].dropna(),
            test[["angle","image_path"]].dropna(),
            test_top[["angle","image_path"]].dropna(),
        ],
        ignore_index=True
    ).drop_duplicates()

    img_sizes = get_angle_img_sizes(union_df, angle_col="angle", path_col="image_path")
    print("img_sizes:", img_sizes)

    train_f = add_formation_features(train, img_sizes=img_sizes)
    test_f  = add_formation_features(test,  img_sizes=img_sizes)
    test_top_f = add_formation_features(test_top, img_sizes=img_sizes)

    # submission order guarantee
    assert len(test_f) == len(test_raw), "row count changed for test!"
    assert len(test_top_f) == len(test_top_raw), "row count changed for test_top!"

    sep("train_f"); show_df(train_f, 3, True)
    sep("test_f"); show_df(test_f, 3, True)

    # ======================
    # 5) Save
    # ======================
    train_f.to_csv(savedir / "train_meta_pp.csv", index=False)
    test_f.to_csv(savedir / "test_meta_pp.csv", index=False)
    test_top_f.to_csv(savedir / "test_top_meta_pp.csv", index=False)

    print("saved:", savedir)

if __name__ == "__main__":
    main()