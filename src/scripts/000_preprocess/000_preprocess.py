import os, warnings
from pathlib import Path
from typing import Any

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
from utils.data import save_config_yaml, dict_to_namespace

warnings.filterwarnings("ignore")


# ===================================
# train(full image) -> crop utils
# ===================================
def get_image_path(row: pd.Series, image_dir: Path) -> Path:
    """train用: {quarter}__{angle}__{session:02d}__{frame:02d}.jpg"""
    fname = f"{row['quarter']}__{row['angle']}__{int(row['session']):02d}__{int(row['frame']):02d}.jpg"
    return Path(image_dir) / fname


def _clip_box_xyxy(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple[int,int,int,int]:
    x1 = int(max(0, min(img_w, x1)))
    x2 = int(max(0, min(img_w, x2)))
    y1 = int(max(0, min(img_h, y1)))
    y2 = int(max(0, min(img_h, y2)))
    return x1, y1, x2, y2


def _safe_imwrite(path: Path, img: np.ndarray, jpeg_quality: int = 95) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    return bool(ok)


def _crop_from_fullimg(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    if img is None:
        return None
    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


def _maybe_resize(img: np.ndarray, out_size: int | None, interpolation: str = "area") -> np.ndarray:
    """
    out_size: None -> resizeしない
    out_size: int  -> (out_size, out_size) にリサイズ
    """
    if out_size is None:
        return img
    out_size = int(out_size)
    if out_size <= 0:
        return img

    interp = str(interpolation).lower()
    if interp == "linear":
        inter = cv2.INTER_LINEAR
    elif interp == "cubic":
        inter = cv2.INTER_CUBIC
    else:
        inter = cv2.INTER_AREA  # downscaleに強い

    return cv2.resize(img, (out_size, out_size), interpolation=inter)


# ===================================
# (NEW) test crop resize worker (TOP-LEVEL)
# ===================================
def _resize_one_test_crop(args: tuple[int, str, str, int, str, int, int, bool]) -> tuple[int, str]:
    """
    Returns: (pos, new_path)
    args:
      pos: int (test_df内の位置)
      in_path: str
      out_dir: str
      out_size: int
      interp: str ("area"/"linear"/"cubic")
      jpeg_quality: int
      orig_idx: int (ファイル名に使う)
      overwrite: bool
    """
    pos, in_path, out_dir, out_size, interp, jpeg_quality, orig_idx, overwrite = args

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    out_path = out_dir_p / f"test_{int(orig_idx)}.jpg"
    if (not overwrite) and out_path.exists():
        return pos, str(out_path.resolve())

    img = cv2.imread(str(in_path))
    if img is None:
        return pos, ""

    interp = str(interp).lower()
    if interp == "linear":
        itp = cv2.INTER_LINEAR
    elif interp == "cubic":
        itp = cv2.INTER_CUBIC
    else:
        itp = cv2.INTER_AREA

    resized = cv2.resize(img, (int(out_size), int(out_size)), interpolation=itp)
    ok = cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        return pos, ""

    return pos, str(out_path.resolve())


def process_single_crop(args: tuple) -> tuple[int, bool, dict]:
    """
    train の1行分bboxを crop して保存する（testは既にcrop配布なので使わない）

    Returns:
      idx, success, info_dict
      info_dict keys:
        - crop_path (normal)
        - crop_path_partial_bottom / crop_keep_partial
        - crop_path_extreme_bottom / crop_keep_extreme
    """
    (
        idx, row, image_dir, output_dir,
        padding_ratio, prefix,
        seed,
        add_bottom_crops,
        bottom_apply_angles,
        bottom_partial_prob, bottom_partial_keep_min, bottom_partial_keep_max,
        bottom_extreme_prob, bottom_extreme_keep_min, bottom_extreme_keep_max,
        jpeg_quality,
        crop_output_size,         # ★追加（Noneならresizeしない）
        crop_interp,              # ★追加
    ) = args

    info = {
        "crop_path": "",
        "crop_path_partial_bottom": "",
        "crop_keep_partial": np.nan,
        "crop_path_extreme_bottom": "",
        "crop_keep_extreme": np.nan,
    }

    try:
        img_path = get_image_path(row, image_dir)
        img = cv2.imread(str(img_path))
        if img is None:
            return idx, False, info

        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        img_h, img_w = img.shape[:2]

        # ---- normal crop (with symmetric padding) ----
        pad_w = int(w * float(padding_ratio))
        pad_h = int(h * float(padding_ratio))

        x1 = x - pad_w
        y1 = y - pad_h
        x2 = x + w + pad_w
        y2 = y + h + pad_h
        x1, y1, x2, y2 = _clip_box_xyxy(x1, y1, x2, y2, img_w, img_h)

        crop = _crop_from_fullimg(img, x1, y1, x2, y2)
        if crop is None:
            return idx, False, info

        crop = _maybe_resize(crop, crop_output_size, crop_interp)

        out_name = f"{prefix}_{idx}.jpg"
        output_path = Path(output_dir) / out_name
        if not _safe_imwrite(output_path, crop, jpeg_quality=jpeg_quality):
            return idx, False, info
        info["crop_path"] = str(output_path.resolve())

        # ---- optional: bottom partial crops (train only) ----
        if add_bottom_crops and (str(row.get("angle", "")) in set(bottom_apply_angles)):
            rng = np.random.RandomState((int(seed) + int(idx)) % (2**32 - 1))

            # partial bottom (e.g. keep 40-70% from bottom)
            if float(bottom_partial_prob) > 0 and rng.rand() < float(bottom_partial_prob):
                keep = float(rng.uniform(bottom_partial_keep_min, bottom_partial_keep_max))
                keep = float(np.clip(keep, 0.01, 0.99))
                y2b = int(min(img_h, y + h))  # no bottom padding
                y1b = int(y2b - max(1, int(h * keep)))
                # x padding only (a bit safer for "foot-only")
                x1b = int(max(0, x - pad_w))
                x2b = int(min(img_w, x + w + pad_w))
                x1b, y1b, x2b, y2b = _clip_box_xyxy(x1b, y1b, x2b, y2b, img_w, img_h)

                crop_b = _crop_from_fullimg(img, x1b, y1b, x2b, y2b)
                if crop_b is not None:
                    crop_b = _maybe_resize(crop_b, crop_output_size, crop_interp)

                    out_name_b = f"{prefix}_pb_{idx}.jpg"
                    out_path_b = Path(output_dir) / out_name_b
                    if _safe_imwrite(out_path_b, crop_b, jpeg_quality=jpeg_quality):
                        info["crop_path_partial_bottom"] = str(out_path_b.resolve())
                        info["crop_keep_partial"] = keep

            # extreme bottom (e.g. keep 10-25% from bottom) -> junk候補
            if float(bottom_extreme_prob) > 0 and rng.rand() < float(bottom_extreme_prob):
                keep = float(rng.uniform(bottom_extreme_keep_min, bottom_extreme_keep_max))
                keep = float(np.clip(keep, 0.01, 0.99))
                y2b = int(min(img_h, y + h))
                y1b = int(y2b - max(1, int(h * keep)))
                x1b = int(max(0, x - pad_w))
                x2b = int(min(img_w, x + w + pad_w))
                x1b, y1b, x2b, y2b = _clip_box_xyxy(x1b, y1b, x2b, y2b, img_w, img_h)

                crop_b = _crop_from_fullimg(img, x1b, y1b, x2b, y2b)
                if crop_b is not None:
                    crop_b = _maybe_resize(crop_b, crop_output_size, crop_interp)

                    out_name_b = f"{prefix}_eb_{idx}.jpg"
                    out_path_b = Path(output_dir) / out_name_b
                    if _safe_imwrite(out_path_b, crop_b, jpeg_quality=jpeg_quality):
                        info["crop_path_extreme_bottom"] = str(out_path_b.resolve())
                        info["crop_keep_extreme"] = keep

        return idx, True, info

    except Exception as e:
        print(f"[crop error] idx={idx} err={e}")
        return idx, False, info


def preprocess_crops(
    df: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    prefix: str,
    padding_ratio: float = 0.10,
    num_workers: int | None = None,
    # bottom crops
    seed: int = 1129,
    add_bottom_crops: bool = False,
    bottom_apply_angles: list[str] | None = None,
    bottom_partial_prob: float = 0.0,
    bottom_partial_keep_min: float = 0.4,
    bottom_partial_keep_max: float = 0.7,
    bottom_extreme_prob: float = 0.0,
    bottom_extreme_keep_min: float = 0.1,
    bottom_extreme_keep_max: float = 0.25,
    jpeg_quality: int = 95,
    # resize
    crop_output_size: int | None = None,
    crop_interp: str = "area",
) -> pd.DataFrame:
    """
    train_meta.csv(相当) df を読み、全bboxを crop して output_dir に保存。
    返り値: orig_idx をキーに crop_path(通常) と追加cropのパス列を返す（行は増やさない）
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    if bottom_apply_angles is None:
        bottom_apply_angles = ["side"]

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["session"] = df["session"].astype(int)
    df["frame"] = df["frame"].astype(int)
    if df["orig_idx"].isna().any():
        raise ValueError("orig_idx has NaN")
    if not df["orig_idx"].is_unique:
        raise ValueError("orig_idx is not unique. crop mapping will break.")

    print(f"[{prefix}] {num_workers} workers, samples={len(df)} -> {output_dir}")
    if add_bottom_crops:
        print(f"[{prefix}] add_bottom_crops=True, angles={bottom_apply_angles}, "
              f"partial_prob={bottom_partial_prob}, extreme_prob={bottom_extreme_prob}")
    if crop_output_size is not None:
        print(f"[{prefix}] resize crop_output_size={int(crop_output_size)} interp={crop_interp}")

    args_list = []
    for _, row in df.iterrows():
        args_list.append((
            int(row["orig_idx"]), row, image_dir, output_dir,
            float(padding_ratio), prefix,
            int(seed),
            bool(add_bottom_crops),
            list(bottom_apply_angles),
            float(bottom_partial_prob), float(bottom_partial_keep_min), float(bottom_partial_keep_max),
            float(bottom_extreme_prob), float(bottom_extreme_keep_min), float(bottom_extreme_keep_max),
            int(jpeg_quality),
            crop_output_size,
            str(crop_interp),
        ))

    # collect
    norm_paths = {}
    pb_paths = {}
    pb_keep = {}
    eb_paths = {}
    eb_keep = {}
    failed = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_crop, args): args[0] for args in args_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"cropping {prefix}"):
            idx, ok, info = fut.result()
            if ok:
                norm_paths[idx] = info.get("crop_path", "")
                if info.get("crop_path_partial_bottom", ""):
                    pb_paths[idx] = info["crop_path_partial_bottom"]
                    pb_keep[idx] = float(info.get("crop_keep_partial", np.nan))
                if info.get("crop_path_extreme_bottom", ""):
                    eb_paths[idx] = info["crop_path_extreme_bottom"]
                    eb_keep[idx] = float(info.get("crop_keep_extreme", np.nan))
            else:
                failed.append(idx)

    out = pd.DataFrame({
        "orig_idx": df["orig_idx"].astype(int),
        "crop_path": df["orig_idx"].map(norm_paths).replace("", np.nan),
        "crop_path_partial_bottom": df["orig_idx"].map(pb_paths).replace("", np.nan),
        "crop_keep_partial": df["orig_idx"].map(pb_keep),
        "crop_path_extreme_bottom": df["orig_idx"].map(eb_paths).replace("", np.nan),
        "crop_keep_extreme": df["orig_idx"].map(eb_keep),
    })

    miss = out["crop_path"].isna().mean()
    print(f"[{prefix}] crop_path missing rate: {miss:.6f}")
    if add_bottom_crops:
        print(f"[{prefix}] partial_bottom generated: {out['crop_path_partial_bottom'].notna().sum()}")
        print(f"[{prefix}] extreme_bottom generated: {out['crop_path_extreme_bottom'].notna().sum()}")
    if failed:
        print(f"[{prefix}] failed idx sample: {failed[:10]}")

    return out


# ===================================
# (NEW) test crop resize
# ===================================
def preprocess_test_crops(
    test_df: pd.DataFrame,
    data_dir: Path,
    output_dir: Path,
    crop_output_size: int,
    crop_interp: str = "area",
    jpeg_quality: int = 95,
    num_workers: int | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    test の配布crop(rel_path)を読み、(crop_output_size,crop_output_size)にリサイズして別保存し、
    test_df["crop_path"] をその保存先に差し替える。

    重要:
      - 失敗した行は元の crop_path を維持（空文字で壊さない）
      - worker はトップレベル関数 _resize_one_test_crop を使用（pickleエラー回避）
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    test_df = test_df.copy()

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 入力パス（元 crop）
    in_paths = test_df["rel_path"].map(lambda p: str((Path(data_dir) / str(p)).resolve())).tolist()

    # 失敗時の保険として、デフォは元の crop_path を使う
    new_paths = test_df["crop_path"].astype(str).tolist()

    args_list = []
    for pos, in_path in enumerate(in_paths):
        orig_idx = int(test_df.iloc[pos]["orig_idx"]) if "orig_idx" in test_df.columns else int(pos)
        args_list.append((
            int(pos),
            str(in_path),
            str(output_dir),
            int(crop_output_size),
            str(crop_interp),
            int(jpeg_quality),
            int(orig_idx),
            bool(overwrite),
        ))

    print(f"[test_resize] workers={num_workers} samples={len(args_list)} -> {output_dir}")
    print(f"[test_resize] size={int(crop_output_size)} interp={crop_interp} jpeg_quality={jpeg_quality}")

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_resize_one_test_crop, a) for a in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="resize test crops"):
            pos, p = fut.result()
            if p:
                new_paths[pos] = p  # 成功時だけ差し替え

    test_df["crop_path"] = new_paths
    miss = (pd.Series(new_paths).isna() | (pd.Series(new_paths) == "")).mean()
    print(f"[test_resize] crop_path missing rate(after): {miss:.6f}")

    return test_df


# ===================================
# (NEW) no-player / background crops
# ===================================
def _xywh_to_xyxy_np(x: np.ndarray, y: np.ndarray, w: np.ndarray, h: np.ndarray) -> np.ndarray:
    x1 = x.astype(np.float32)
    y1 = y.astype(np.float32)
    x2 = (x + w).astype(np.float32)
    y2 = (y + h).astype(np.float32)
    return np.stack([x1, y1, x2, y2], axis=1)

def _iou_1_to_n(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)

    xA = np.maximum(box[0], boxes[:, 0])
    yA = np.maximum(box[1], boxes[:, 1])
    xB = np.minimum(box[2], boxes[:, 2])
    yB = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(xB - xA, 0.0, None)
    inter_h = np.clip(yB - yA, 0.0, None)
    inter = inter_w * inter_h

    area_box = np.clip(box[2] - box[0], 0.0, None) * np.clip(box[3] - box[1], 0.0, None)
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)

    union = area_box + area_boxes - inter + 1e-9
    return (inter / union).astype(np.float32)

def _clip_xyxy_np(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, img_w))
    x2 = float(np.clip(x2, 0, img_w))
    y1 = float(np.clip(y1, 0, img_h))
    y2 = float(np.clip(y2, 0, img_h))
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def build_bg_bboxes(
    train_df: pd.DataFrame,
    img_sizes: dict,
    *,
    padding_ratio: float,
    seed: int = 1129,
    apply_angles: list[str] | None = None,
    per_frame: int = 1,
    iou_thr: float = 0.0,      # padding後boxのmax IoUがこれ以下なら採用
    max_trials: int = 50,
    wh_scale_min: float = 0.8,
    wh_scale_max: float = 1.2,
    min_wh: int = 16,
) -> pd.DataFrame:
    if apply_angles is None:
        apply_angles = ["top", "side"]

    rng = np.random.default_rng(int(seed))

    wh_pool = {}
    for ang in apply_angles:
        sub = train_df[train_df["angle"] == ang]
        if len(sub) == 0:
            continue
        wh_pool[ang] = sub[["w", "h"]].to_numpy()

    group_cols = ["quarter", "session", "frame", "angle"]
    rows = []
    next_idx = int(train_df["orig_idx"].max()) + 1

    for (q, s, f, ang), g in train_df.groupby(group_cols, sort=False):
        if str(ang) not in apply_angles:
            continue
        if ang not in img_sizes:
            continue
        if ang not in wh_pool:
            continue

        img_w, img_h = img_sizes[ang]
        gt_xyxy = _xywh_to_xyxy_np(
            g["x"].to_numpy(), g["y"].to_numpy(),
            g["w"].to_numpy(), g["h"].to_numpy(),
        )

        for k in range(int(per_frame)):
            found = False
            best_max_iou = None
            best_xywh = None

            for _ in range(int(max_trials)):
                w0, h0 = wh_pool[ang][rng.integers(0, len(wh_pool[ang]))]
                scale = float(rng.uniform(float(wh_scale_min), float(wh_scale_max)))
                w = int(max(int(min_wh), round(float(w0) * scale)))
                h = int(max(int(min_wh), round(float(h0) * scale)))

                w = int(min(w, img_w - 1))
                h = int(min(h, img_h - 1))
                if w <= 1 or h <= 1:
                    continue

                x = int(rng.integers(0, img_w - w))
                y = int(rng.integers(0, img_h - h))

                pad_w = int(w * float(padding_ratio))
                pad_h = int(h * float(padding_ratio))
                box_pad = np.array([x - pad_w, y - pad_h, x + w + pad_w, y + h + pad_h], dtype=np.float32)
                box_pad = _clip_xyxy_np(box_pad, img_w, img_h)

                ious = _iou_1_to_n(box_pad, gt_xyxy)
                max_iou = float(ious.max()) if len(ious) else 0.0

                if (best_max_iou is None) or (max_iou < best_max_iou):
                    best_max_iou = max_iou
                    best_xywh = (x, y, w, h)

                if max_iou <= float(iou_thr):
                    found = True
                    break

            if found and best_xywh is not None:
                x, y, w, h = best_xywh
                rows.append({
                    "orig_idx": int(next_idx),
                    "quarter": q,
                    "angle": ang,
                    "session": int(s),
                    "frame": int(f),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "label_id": -1,
                    "bg_k": int(k),
                    "bg_max_iou_pad": float(best_max_iou if best_max_iou is not None else 0.0),
                })
                next_idx += 1

    return pd.DataFrame(rows)


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

    if Q[:, 0].max() < -Q[:, 0].min():
        Q *= -1.0

    return Q.astype(np.float32), float(s)


def _compute_formation_from_P(P: np.ndarray) -> dict[str, np.ndarray]:
    n = int(P.shape[0])
    Qn, shape_scale = normalize_points_rigid(P)

    center = P.mean(axis=0)
    dist_center = np.sqrt(((P - center) ** 2).sum(axis=1)).astype(np.float32)

    rank_x = (pd.Series(P[:, 0]).rank(method="first").to_numpy(np.int32) - 1).astype(np.float32)
    rank_y = (pd.Series(P[:, 1]).rank(method="first").to_numpy(np.int32) - 1).astype(np.float32)

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

    return {
        "n_players": np.full((n,), n, dtype=np.int32),
        "fx": Qn[:, 0].astype(np.float32),
        "fy": Qn[:, 1].astype(np.float32),
        "dist_center": dist_center,
        "rank_x": rank_x.astype(np.float32),
        "rank_y": rank_y.astype(np.float32),
        "rank_x_norm": rank_x_norm,
        "rank_y_norm": rank_y_norm,
        "nn_dist": nn_dist,
        "mean_dist": mean_dist,
        "shape_scale": np.full((n,), np.float32(shape_scale), dtype=np.float32),
    }


def _build_P_for_group(g: pd.DataFrame, img_sizes: dict) -> np.ndarray:
    n = len(g)
    P = np.zeros((n, 2), dtype=np.float32)
    g = g.copy()
    g["__pos"] = np.arange(n, dtype=np.int32)

    for ang, gg in g.groupby("angle", sort=False):
        if ang not in img_sizes:
            continue
        img_w, img_h = img_sizes[ang]
        P_ang = bbox_anchor_xy(gg, img_w, img_h)
        P[gg["__pos"].to_numpy()] = P_ang
    return P


def add_formation_features(
    df: pd.DataFrame,
    img_sizes: dict,
    key_cols=("quarter","session","frame"),
    # augmentation (train only)
    aug_enable: bool = False,
    aug_prob: float = 0.0,
    aug_target_n_mode: str = "uniform",
    aug_target_n_min: int = 8,
    aug_target_n_max: int = 14,
    target_n_choices: np.ndarray | None = None,
    target_n_probs: np.ndarray | None = None,
    dummy_jitter_std: float = 0.01,
    dummy_jitter_clip: float = 0.05,
    seed: int = 1129,
) -> pd.DataFrame:
    df = df.copy()
    if "__row_id" not in df.columns:
        df["__row_id"] = np.arange(len(df), dtype=np.int64)

    rng = np.random.default_rng(int(seed))

    if aug_target_n_mode == "from_test":
        if target_n_choices is None or target_n_probs is None:
            raise ValueError("aug_target_n_mode='from_test' requires target_n_choices/probs")
        target_n_choices = np.asarray(target_n_choices, dtype=int)
        target_n_probs = np.asarray(target_n_probs, dtype=float)
        target_n_probs = target_n_probs / target_n_probs.sum()

        def sample_target_n() -> int:
            return int(rng.choice(target_n_choices, p=target_n_probs))
    else:
        def sample_target_n() -> int:
            return int(rng.integers(int(aug_target_n_min), int(aug_target_n_max) + 1))

    out_chunks = []
    for _, g in df.groupby(list(key_cols), sort=False):
        g = g.copy()
        n = len(g)
        if n == 0:
            continue

        P = _build_P_for_group(g, img_sizes)
        feats = _compute_formation_from_P(P)

        gg_out = g.copy()
        gg_out["n_players"] = feats["n_players"].astype(np.int32)
        gg_out["fx"] = feats["fx"]
        gg_out["fy"] = feats["fy"]
        gg_out["dist_center"] = feats["dist_center"]
        gg_out["rank_x"] = feats["rank_x"].astype(np.int16)
        gg_out["rank_y"] = feats["rank_y"].astype(np.int16)
        gg_out["rank_x_norm"] = feats["rank_x_norm"]
        gg_out["rank_y_norm"] = feats["rank_y_norm"]
        gg_out["nn_dist"] = feats["nn_dist"]
        gg_out["mean_dist"] = feats["mean_dist"]
        gg_out["shape_scale"] = feats["shape_scale"].astype(np.float32)

        bbox_area = (gg_out["w"].to_numpy(np.float32) * gg_out["h"].to_numpy(np.float32))
        gg_out["bbox_area"] = np.log1p(bbox_area)
        gg_out["bbox_aspect"] = (gg_out["w"].to_numpy(np.float32) / (gg_out["h"].to_numpy(np.float32) + 1e-6))

        gg_out["formation_variant"] = "base"
        gg_out["formation_target_n"] = n
        gg_out["formation_dummy_n"] = 0
        gg_out["__aug_rank"] = 0
        out_chunks.append(gg_out)

        if (not aug_enable) or (float(aug_prob) <= 0):
            continue
        if rng.random() >= float(aug_prob):
            continue

        target_n = sample_target_n()
        target_n = int(np.clip(target_n, 1, 10_000))
        if target_n == n:
            continue

        if target_n < n:
            k = int(max(1, target_n))
            sel = rng.choice(n, size=k, replace=False)
            sel = np.sort(sel)
            g_sub = g.iloc[sel].copy()
            P_sub = _build_P_for_group(g_sub, img_sizes)
            feats_sub = _compute_formation_from_P(P_sub)

            out_sub = g_sub.copy()
            out_sub["n_players"] = feats_sub["n_players"].astype(np.int32)
            out_sub["fx"] = feats_sub["fx"]
            out_sub["fy"] = feats_sub["fy"]
            out_sub["dist_center"] = feats_sub["dist_center"]
            out_sub["rank_x"] = feats_sub["rank_x"].astype(np.int16)
            out_sub["rank_y"] = feats_sub["rank_y"].astype(np.int16)
            out_sub["rank_x_norm"] = feats_sub["rank_x_norm"]
            out_sub["rank_y_norm"] = feats_sub["rank_y_norm"]
            out_sub["nn_dist"] = feats_sub["nn_dist"]
            out_sub["mean_dist"] = feats_sub["mean_dist"]
            out_sub["shape_scale"] = feats_sub["shape_scale"].astype(np.float32)

            bbox_area = (out_sub["w"].to_numpy(np.float32) * out_sub["h"].to_numpy(np.float32))
            out_sub["bbox_area"] = np.log1p(bbox_area)
            out_sub["bbox_aspect"] = (out_sub["w"].to_numpy(np.float32) / (out_sub["h"].to_numpy(np.float32) + 1e-6))

            out_sub["formation_variant"] = "reduce"
            out_sub["formation_target_n"] = int(target_n)
            out_sub["formation_dummy_n"] = int(n - k)
            out_sub["__aug_rank"] = 1
            out_chunks.append(out_sub)

        else:
            dummy_n = int(target_n - n)
            if dummy_n <= 0:
                continue

            base_idx = rng.integers(0, n, size=dummy_n)
            P_dummy = P[base_idx].copy()
            noise = rng.normal(0.0, float(dummy_jitter_std), size=P_dummy.shape).astype(np.float32)
            if float(dummy_jitter_clip) > 0:
                noise = np.clip(noise, -float(dummy_jitter_clip), float(dummy_jitter_clip))
            P_dummy = np.clip(P_dummy + noise, 0.0, 1.0)

            P_tot = np.concatenate([P, P_dummy], axis=0)
            feats_tot = _compute_formation_from_P(P_tot)

            out_inc = g.copy()
            out_inc["n_players"] = int(target_n)
            out_inc["fx"] = feats_tot["fx"][:n]
            out_inc["fy"] = feats_tot["fy"][:n]
            out_inc["dist_center"] = feats_tot["dist_center"][:n]
            out_inc["rank_x"] = feats_tot["rank_x"][:n].astype(np.int16)
            out_inc["rank_y"] = feats_tot["rank_y"][:n].astype(np.int16)
            out_inc["rank_x_norm"] = feats_tot["rank_x_norm"][:n]
            out_inc["rank_y_norm"] = feats_tot["rank_y_norm"][:n]
            out_inc["nn_dist"] = feats_tot["nn_dist"][:n]
            out_inc["mean_dist"] = feats_tot["mean_dist"][:n]
            out_inc["shape_scale"] = feats_tot["shape_scale"][:n].astype(np.float32)

            bbox_area = (out_inc["w"].to_numpy(np.float32) * out_inc["h"].to_numpy(np.float32))
            out_inc["bbox_area"] = np.log1p(bbox_area)
            out_inc["bbox_aspect"] = (out_inc["w"].to_numpy(np.float32) / (out_inc["h"].to_numpy(np.float32) + 1e-6))

            out_inc["formation_variant"] = "increase"
            out_inc["formation_target_n"] = int(target_n)
            out_inc["formation_dummy_n"] = int(dummy_n)
            out_inc["__aug_rank"] = 2
            out_chunks.append(out_inc)

    out = pd.concat(out_chunks, axis=0, ignore_index=True)
    out = out.sort_values(["__row_id", "__aug_rank"]).drop(columns=["__aug_rank", "__row_id"]).reset_index(drop=True)
    return out


def _expand_with_bottom_crops(train_df: pd.DataFrame, apply_to_all_formation: bool = False) -> pd.DataFrame:
    df = train_df.copy()

    df["crop_variant"] = "normal"
    df["crop_keep_ratio"] = 1.0
    df["is_junk"] = 0

    if "formation_variant" in df.columns and (not apply_to_all_formation):
        base_mask = (df["formation_variant"] == "base")
    else:
        base_mask = np.ones(len(df), dtype=bool)

    aug_rows = []

    if "crop_path_partial_bottom" in df.columns:
        m = base_mask & df["crop_path_partial_bottom"].notna()
        if m.any():
            tmp = df.loc[m].copy()
            tmp["crop_path"] = tmp["crop_path_partial_bottom"]
            tmp["crop_variant"] = "partial_bottom"
            tmp["crop_keep_ratio"] = tmp.get("crop_keep_partial", np.nan)
            tmp["is_junk"] = 0
            aug_rows.append(tmp)

    if "crop_path_extreme_bottom" in df.columns:
        m = base_mask & df["crop_path_extreme_bottom"].notna()
        if m.any():
            tmp = df.loc[m].copy()
            tmp["crop_path"] = tmp["crop_path_extreme_bottom"]
            tmp["crop_variant"] = "extreme_bottom"
            tmp["crop_keep_ratio"] = tmp.get("crop_keep_extreme", np.nan)
            tmp["is_junk"] = 1
            aug_rows.append(tmp)

    if aug_rows:
        out = pd.concat([df] + aug_rows, axis=0, ignore_index=True)
    else:
        out = df

    drop_cols = [c for c in [
        "crop_path_partial_bottom", "crop_keep_partial",
        "crop_path_extreme_bottom", "crop_keep_extreme",
    ] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

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

    # resize settings
    crop_output_size = getattr(config, "crop_output_size", None)  # None or int
    crop_interp = str(getattr(config, "crop_interp", "area"))
    # 互換: crop_jpeg_quality が無ければ bottom_crop_jpeg_quality を見に行く
    crop_jpeg_quality = int(getattr(config, "crop_jpeg_quality", getattr(config, "bottom_crop_jpeg_quality", 95)))

    # crop dirs
    CROP_ROOT = Path(config.crop_dir).resolve()
    if crop_output_size is None:
        CROP_TRAIN = CROP_ROOT / "train"
    else:
        CROP_TRAIN = CROP_ROOT / f"train_sz{int(crop_output_size)}"

    # ======================
    # 1) Read raw meta
    # ======================
    train_raw = pd.read_csv(DATA_DIR / "train_meta.csv")
    test_raw  = pd.read_csv(DATA_DIR / "test_meta.csv")

    train_raw["session"] = train_raw["session"].astype(int)
    train_raw["frame"] = train_raw["frame"].astype(int)

    test_raw["session"] = test_raw["session_no"].astype(int)
    test_raw["frame"] = test_raw["frame_in_session"].astype(int)

    train_raw["orig_idx"] = np.arange(len(train_raw), dtype=np.int64)
    test_raw["orig_idx"] = np.arange(len(test_raw), dtype=np.int64)

    # test: rel_path を保持（crop_path は後で差し替える可能性あり）
    test_raw["rel_path"] = test_raw["rel_path"].astype(str)
    test_raw["crop_path"] = test_raw["rel_path"].map(lambda p: str((DATA_DIR / p).resolve()))

    # test crop をリサイズして別保存する（指定があれば）
    if crop_output_size is not None:
        test_raw = preprocess_test_crops(
            test_df=test_raw,
            data_dir=DATA_DIR,
            output_dir=CROP_ROOT / f"test_sz{int(crop_output_size)}",
            crop_output_size=int(crop_output_size),
            crop_interp=crop_interp,
            jpeg_quality=crop_jpeg_quality,
            num_workers=None,
            overwrite=False,
        )

    # ======================
    # 2) full image path（img size取得にも使う）
    # ======================
    train = train_raw.copy()
    train["image_path"] = train.apply(lambda r: str(get_image_path(r, IMAGE_DIR).resolve()), axis=1)
    train["paired_path"] = pd.NA

    test = test_raw.copy()
    test["image_path"] = pd.NA
    test["paired_path"] = pd.NA

    # ======================
    # 3) train crops only（testは作らない：ただし test は resize 済み crop_path を持つ）
    # ======================
    if config.use_crops:
        crop_map = preprocess_crops(
            df=train,
            image_dir=IMAGE_DIR,
            output_dir=CROP_TRAIN,
            prefix="train",
            padding_ratio=float(config.padding_ratio),
            num_workers=None,
            seed=int(config.seed),
            add_bottom_crops=bool(getattr(config, "add_bottom_crops", False)),
            bottom_apply_angles=list(getattr(config, "bottom_crop_apply_angles", ["side"])),
            bottom_partial_prob=float(getattr(config, "bottom_partial_prob", 0.0)),
            bottom_partial_keep_min=float(getattr(config, "bottom_partial_keep_min", 0.4)),
            bottom_partial_keep_max=float(getattr(config, "bottom_partial_keep_max", 0.7)),
            bottom_extreme_prob=float(getattr(config, "bottom_extreme_prob", 0.0)),
            bottom_extreme_keep_min=float(getattr(config, "bottom_extreme_keep_min", 0.1)),
            bottom_extreme_keep_max=float(getattr(config, "bottom_extreme_keep_max", 0.25)),
            jpeg_quality=crop_jpeg_quality,
            crop_output_size=crop_output_size,
            crop_interp=crop_interp,
        )
        train = train.merge(crop_map, on="orig_idx", how="left")
        train["crop_path"] = train["crop_path"].astype("object")
    else:
        train["crop_path"] = train["image_path"]

    # ======================
    # 4) formation features
    # ======================
    img_sizes = get_angle_img_sizes_from_train(train, IMAGE_DIR)
    print("img_sizes:", img_sizes)

    test_np = test.groupby(["quarter","session","frame","angle"], sort=False).size()
    print("[test] n_players dist:\n", test_np.value_counts().sort_index())

    target_n_choices = None
    target_n_probs = None
    if str(getattr(config, "formation_aug_target_n_mode", "uniform")) == "from_test":
        vc = test_np.value_counts().sort_index()
        vc = vc[(vc.index >= config.formation_aug_target_n_min) & (vc.index <= config.formation_aug_target_n_max)]
        target_n_choices = vc.index.to_numpy().astype(int)
        target_n_probs = (vc / vc.sum()).to_numpy().astype(float)
        print("[formation_aug] target_n from test:", dict(zip(target_n_choices.tolist(), target_n_probs.tolist())))

    train_f = add_formation_features(
        train,
        img_sizes=img_sizes,
        key_cols=("quarter","session","frame","angle"),
        aug_enable=bool(getattr(config, "formation_aug_enable", False)),
        aug_prob=float(getattr(config, "formation_aug_prob", 0.0)),
        aug_target_n_mode=str(getattr(config, "formation_aug_target_n_mode", "uniform")),
        aug_target_n_min=int(getattr(config, "formation_aug_target_n_min", 8)),
        aug_target_n_max=int(getattr(config, "formation_aug_target_n_max", 14)),
        target_n_choices=target_n_choices,
        target_n_probs=target_n_probs,
        dummy_jitter_std=float(getattr(config, "formation_dummy_jitter_std", 0.01)),
        dummy_jitter_clip=float(getattr(config, "formation_dummy_jitter_clip", 0.05)),
        seed=int(config.seed),
    )

    test_f = add_formation_features(
        test,
        img_sizes=img_sizes,
        key_cols=("quarter","session","frame","angle"),
        aug_enable=False,
        seed=int(config.seed),
    )

    # ======================
    # (NEW) no-player / background crops (train only)
    # ======================
    bg_df = None
    add_bg_crops = bool(getattr(config, "add_bg_crops", False))

    if add_bg_crops:
        if not bool(config.use_crops):
            print("[bg] WARN: use_crops=False なので bg crop はスキップ（bbox crop が作れない）")
        else:
            bg_apply_angles = list(getattr(config, "bg_apply_angles", ["top", "side"]))
            bg_per_frame = int(getattr(config, "bg_per_frame", 1))
            bg_iou_thr = float(getattr(config, "bg_iou_thr", 0.0))
            bg_max_trials = int(getattr(config, "bg_max_trials", 50))
            bg_wh_scale_min = float(getattr(config, "bg_wh_scale_min", 0.8))
            bg_wh_scale_max = float(getattr(config, "bg_wh_scale_max", 1.2))
            bg_min_wh = int(getattr(config, "bg_min_wh", 16))

            print(f"[bg] generating bg boxes: per_frame={bg_per_frame}, iou_thr(pad)={bg_iou_thr}, angles={bg_apply_angles}")
            bg_df = build_bg_bboxes(
                train_df=train,
                img_sizes=img_sizes,
                padding_ratio=float(config.padding_ratio),
                seed=int(config.seed),
                apply_angles=bg_apply_angles,
                per_frame=bg_per_frame,
                iou_thr=bg_iou_thr,
                max_trials=bg_max_trials,
                wh_scale_min=bg_wh_scale_min,
                wh_scale_max=bg_wh_scale_max,
                min_wh=bg_min_wh,
            )
            print(f"[bg] generated rows: {len(bg_df)}")

            if len(bg_df) > 0:
                bg_df["image_path"] = bg_df.apply(lambda r: str(get_image_path(r, IMAGE_DIR).resolve()), axis=1)
                bg_df["paired_path"] = pd.NA

                CROP_TRAIN_BG = CROP_TRAIN / "bg"
                crop_map_bg = preprocess_crops(
                    df=bg_df,
                    image_dir=IMAGE_DIR,
                    output_dir=CROP_TRAIN_BG,
                    prefix="bg",
                    padding_ratio=float(config.padding_ratio),
                    num_workers=None,
                    seed=int(config.seed),
                    add_bottom_crops=False,
                    jpeg_quality=crop_jpeg_quality,
                    crop_output_size=crop_output_size,
                    crop_interp=crop_interp,
                )
                bg_df = bg_df.merge(crop_map_bg, on="orig_idx", how="left")
                bg_df["crop_path"] = bg_df["crop_path"].astype("object")

                bg_df.to_csv(savedir / "train_bg_meta.csv", index=False)
                print("[bg] saved debug:", savedir / "train_bg_meta.csv")

    # ======================
    # 5) expand rows by extra bottom crops (train only)
    # ======================
    if bool(getattr(config, "add_bottom_crops", False)):
        train_f = _expand_with_bottom_crops(
            train_f,
            apply_to_all_formation=bool(getattr(config, "bottom_crop_apply_to_all_formation", False)),
        )

    # ======================
    # append bg rows to train_f (after bottom expand)
    # ======================
    if bg_df is not None and len(bg_df) > 0:
        tmp = pd.concat([train, bg_df], axis=0, ignore_index=True)

        tmp_f = add_formation_features(
            tmp,
            img_sizes=img_sizes,
            key_cols=("quarter","session","frame","angle"),
            aug_enable=False,
            seed=int(config.seed),
        )
        bg_f = tmp_f[tmp_f["label_id"].astype(int) == -1].copy()

        drop_cols = [c for c in [
            "crop_path_partial_bottom", "crop_keep_partial",
            "crop_path_extreme_bottom", "crop_keep_extreme",
        ] if c in bg_f.columns]
        if drop_cols:
            bg_f = bg_f.drop(columns=drop_cols)

        bg_f["crop_variant"] = str(getattr(config, "bg_crop_variant", "bg_random"))
        bg_f["crop_keep_ratio"] = 1.0
        bg_f["is_junk"] = 1 if bool(getattr(config, "bg_mark_as_junk", True)) else 0

        train_f = pd.concat([train_f, bg_f], axis=0, ignore_index=True)
        print("[bg] appended to train_f:", len(bg_f), "rows")

    # ======================
    # 6) Save
    # ======================
    train_f.to_csv(savedir / "train_meta_pp.csv", index=False)
    test_f.to_csv(savedir / "test_meta_pp.csv", index=False)

    print("saved:", savedir.resolve())
    print("[paths]")
    print("  train_meta_pp:", (savedir / "train_meta_pp.csv").resolve())
    print("  test_meta_pp :", (savedir / "test_meta_pp.csv").resolve())
    print("  crop_train_dir:", CROP_TRAIN.resolve())
    if crop_output_size is not None:
        print("  crop_test_dir :", (CROP_ROOT / f"test_sz{int(crop_output_size)}").resolve())

    if "n_players" in train_f.columns:
        print("[train] n_players dist:\n", train_f["n_players"].value_counts().sort_index())
    if "crop_variant" in train_f.columns:
        print("[train] crop_variant dist:\n", train_f["crop_variant"].value_counts())
    if "formation_variant" in train_f.columns:
        print("[train] formation_variant dist:\n", train_f["formation_variant"].value_counts())


if __name__ == "__main__":
    main()