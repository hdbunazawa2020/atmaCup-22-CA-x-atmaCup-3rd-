import os, warnings
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
# train(full image) -> crop utils
# ===================================
def get_image_path(row: pd.Series, image_dir: Path) -> Path:
    """train用: {quarter}__{angle}__{session:02d}__{frame:02d}.jpg"""
    fname = f"{row['quarter']}__{row['angle']}__{int(row['session']):02d}__{int(row['frame']):02d}.jpg"
    return Path(image_dir) / fname


def process_single_crop(args: tuple) -> tuple[int, bool, str]:
    """
    trainの1行分bboxを crop して保存する（testは既にcrop配布なので使わない）
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

        # 衝突回避 prefix
        out_name = f"{prefix}_{idx}.jpg"
        output_path = Path(output_dir) / out_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return idx, True, str(output_path.resolve())

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
    train_meta.csv を読み、全bboxを crop して output_dir に保存。
    返り値: 元csv + crop_path列（失敗はNaN）
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    output_dir = Path(output_dir).resolve()
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
# formation features
# ===================================
def get_img_size_from_path(path: str):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"image not found: {path}")
    h, w = img.shape[:2]
    return w, h


def get_angle_img_sizes_from_train(train_df: pd.DataFrame, image_dir: Path) -> dict:
    """
    testはフル画像が無いので、img_w/img_h は train のフル画像から angle別に1枚だけ取得する。
    """
    sizes = {}
    for ang in ["top", "side"]:
        sub = train_df[train_df["angle"] == ang]
        if len(sub) == 0:
            continue
        p = get_image_path(sub.iloc[0], image_dir)
        sizes[ang] = get_img_size_from_path(str(p))
    if not sizes:
        raise RuntimeError("failed to infer image sizes from train")
    return sizes


def bbox_anchor_xy(df, img_w, img_h):
    """
    top : bbox center
    side: bbox bottom-center（足元proxy）
    """
    x = df["x"].to_numpy(np.float32)
    y = df["y"].to_numpy(np.float32)
    w = df["w"].to_numpy(np.float32)
    h = df["h"].to_numpy(np.float32)

    cx = x + 0.5 * w
    cy = np.where(df["angle"].to_numpy() == "top", y + 0.5*h, y + h)

    nx = cx / float(img_w)
    ny = cy / float(img_h)
    return np.stack([nx, ny], axis=1).astype(np.float32)


def normalize_points_rigid(P, eps=1e-6):
    """
    translation/scale/rotation を（ざっくり）除去するPCA正規化。
    """
    c = P.mean(axis=0, keepdims=True)
    X = P - c
    s = np.sqrt((X**2).sum(axis=1).mean()) + eps
    X = X / s

    C = (X.T @ X) / X.shape[0]
    _, eigvecs = np.linalg.eigh(C)
    v1 = eigvecs[:, 1]
    v2 = eigvecs[:, 0]
    R = np.stack([v1, v2], axis=1)
    Q = X @ R

    # PCA符号の不定性を固定（180度反転対策）
    if Q[:, 0].max() < -Q[:, 0].min():
        Q *= -1.0

    return Q.astype(np.float32), float(s)


def add_formation_features(df: pd.DataFrame, img_sizes: dict, key_cols=("quarter","session","frame")):
    """
    ※リスタート後testでも frame単位で8〜14 bbox があるので formation は計算可能
    ただし N(=n_players) が可変なので、n_players と rank正規化を追加するのが重要。
    """
    df = df.copy()
    df["__row_id"] = np.arange(len(df), dtype=np.int64)

    out_chunks = []
    for _, g in df.groupby(list(key_cols), sort=False):
        g = g.copy()
        n = len(g)
        g["n_players"] = n  # ★重要：そのフレームに何bboxあるか

        g["__pos"] = np.arange(n, dtype=np.int32)
        P = np.zeros((n, 2), dtype=np.float32)

        for ang, gg in g.groupby("angle", sort=False):
            if ang not in img_sizes:
                # 通常は起きないが、念のため
                continue
            img_w, img_h = img_sizes[ang]
            P_ang = bbox_anchor_xy(gg, img_w, img_h)
            P[gg["__pos"].to_numpy()] = P_ang

        Qn, shape_scale = normalize_points_rigid(P)

        center = P.mean(axis=0)
        dist_center = np.sqrt(((P - center) ** 2).sum(axis=1)).astype(np.float32)

        rank_x = (pd.Series(P[:, 0]).rank(method="first").to_numpy(np.int32) - 1).astype(np.float32)
        rank_y = (pd.Series(P[:, 1]).rank(method="first").to_numpy(np.int32) - 1).astype(np.float32)

        # rank を Nで正規化（Nが8〜14で揺れるので重要）
        denom = max(1, n - 1)
        rank_x_norm = (rank_x / denom).astype(np.float32)
        rank_y_norm = (rank_y / denom).astype(np.float32)

        if n >= 2:
            D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)).astype(np.float32)
            np.fill_diagonal(D, np.nan)
            nn_dist = np.nanmin(D, axis=1).astype(np.float32)
            mean_dist = np.nanmean(D, axis=1).astype(np.float32)
        else:
            nn_dist = np.full((n,), np.nan, dtype=np.float32)
            mean_dist = np.full((n,), np.nan, dtype=np.float32)

        gg_out = g.drop(columns="__pos").copy()
        gg_out["fx"] = Qn[:, 0]
        gg_out["fy"] = Qn[:, 1]
        gg_out["dist_center"] = dist_center
        gg_out["rank_x"] = rank_x.astype(np.int16)
        gg_out["rank_y"] = rank_y.astype(np.int16)
        gg_out["rank_x_norm"] = rank_x_norm
        gg_out["rank_y_norm"] = rank_y_norm
        gg_out["nn_dist"] = nn_dist
        gg_out["mean_dist"] = mean_dist
        gg_out["shape_scale"] = np.float32(shape_scale)

        bbox_area = (gg_out["w"].to_numpy(np.float32) * gg_out["h"].to_numpy(np.float32))
        gg_out["bbox_area"] = np.log1p(bbox_area)  # ★log圧縮推奨
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

    DATA_DIR  = Path(config.data_dir).resolve()
    IMAGE_DIR = Path(config.image_dir).resolve()

    # crops dir（ここは root をそのまま使う）
    CROP_ROOT  = Path(config.crop_dir).resolve()
    CROP_TRAIN = CROP_ROOT / "train"

    # ======================
    # 1) Read raw meta
    # ======================
    train_raw = pd.read_csv(DATA_DIR / "train_meta.csv")
    test_raw  = pd.read_csv(DATA_DIR / "test_meta.csv")

    # trainは session/frame がある
    train_raw["session"] = train_raw["session"].astype(int)
    train_raw["frame"] = train_raw["frame"].astype(int)

    # testは session_no / frame_in_session を session/frame に寄せる（あなたの確認どおり存在する）
    test_raw["session"] = test_raw["session_no"].astype(int)
    test_raw["frame"] = test_raw["frame_in_session"].astype(int)

    # testは配布cropを使う（rel_path）
    test_raw["crop_path"] = test_raw["rel_path"].map(lambda p: str((DATA_DIR / p).resolve()))

    # ======================
    # 2) train: full image path（img size取得にも使う）
    # ======================
    train = train_raw.copy()
    train["image_path"] = train.apply(lambda r: str(get_image_path(r, IMAGE_DIR).resolve()), axis=1)
    # paired_path はあっても良いが、必須ではないので今回は省略（必要なら復活）
    train["paired_path"] = pd.NA

    # testはフル画像無しなので image_path は NA のまま
    test = test_raw.copy()
    test["image_path"] = pd.NA
    test["paired_path"] = pd.NA

    # ======================
    # 3) train crops only（testは作らない）
    # ======================
    if config.use_crops:
        train_crop_df = preprocess_crops(
            csv_path=DATA_DIR / "train_meta.csv",
            image_dir=IMAGE_DIR,
            output_dir=CROP_TRAIN,
            prefix="train",
            padding_ratio=float(config.padding_ratio),
            num_workers=None,
        )
        assert len(train_crop_df) == len(train), "train row mismatch"
        train["crop_path"] = train_crop_df["crop_path"].values
    else:
        # 非推奨：学習側でフル画像+ bbox crop が必要になる
        train["crop_path"] = train["image_path"]

    # ======================
    # 4) formation features（train/test両方に付与可能）
    #    img_sizesは train のフル画像から推定する
    # ======================
    img_sizes = get_angle_img_sizes_from_train(train, IMAGE_DIR)
    print("img_sizes:", img_sizes)

    train_f = add_formation_features(train, img_sizes=img_sizes)
    test_f  = add_formation_features(test,  img_sizes=img_sizes)

    # ======================
    # 5) Save
    # ======================
    train_f.to_csv(savedir / "train_meta_pp.csv", index=False)
    test_f.to_csv(savedir / "test_meta_pp.csv", index=False)

    print("saved:", savedir.resolve())


if __name__ == "__main__":
    main()