from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, accuracy_score

import sys
sys.path.append("..")  # src/scripts から実行想定
from utils.data import save_config_yaml, dict_to_namespace

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")


# ========================
# scoring
# ========================
def _labels_with_unknown(y: np.ndarray, pred: np.ndarray) -> list[int]:
    # -1 を含めて評価（macro_f1のlabelsに出現クラスだけ入れる）
    labels = np.unique(np.concatenate([y, pred])).tolist()
    return labels


def score_with_unknown_1d(
    y_true: np.ndarray,
    pred_class: np.ndarray,
    max_sim: np.ndarray,
    thr_sim: float,
    unknown_ids: list[int] | None = None,
    metric: str = "macro_f1",
) -> float:
    """
    1D: max_sim < thr_sim を unknown(-1) にする
    """
    y = y_true.copy()
    if unknown_ids:
        y = np.where(np.isin(y, unknown_ids), -1, y)

    pred = pred_class.copy()
    pred = np.where(max_sim < thr_sim, -1, pred)

    labels = _labels_with_unknown(y, pred)
    if metric == "macro_f1":
        return float(f1_score(y, pred, average="macro", labels=labels))
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    raise ValueError(f"unknown metric: {metric}")


def score_with_unknown_2d(
    y_true: np.ndarray,
    pred_class: np.ndarray,
    max_sim: np.ndarray,
    second_sim: np.ndarray,
    thr_sim: float,
    thr_margin: float,
    unknown_ids: list[int] | None = None,
    metric: str = "macro_f1",
    combine_mode: str = "or",   # "or" or "and"
) -> float:
    """
    2D: max_sim と margin の閾値で unknown(-1) を判定
      - combine_mode="or":  (max_sim < thr_sim) OR (margin < thr_margin) -> unknown
      - combine_mode="and": (max_sim < thr_sim) AND (margin < thr_margin) -> unknown
    """
    y = y_true.copy()
    if unknown_ids:
        y = np.where(np.isin(y, unknown_ids), -1, y)

    pred = pred_class.copy()
    margin = max_sim - second_sim

    cond_sim = (max_sim < thr_sim)
    cond_m   = (margin < thr_margin)

    if combine_mode == "or":
        unk = cond_sim | cond_m
    elif combine_mode == "and":
        unk = cond_sim & cond_m
    else:
        raise ValueError(f"unknown combine_mode: {combine_mode}")

    pred = np.where(unk, -1, pred)

    labels = _labels_with_unknown(y, pred)
    if metric == "macro_f1":
        return float(f1_score(y, pred, average="macro", labels=labels))
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    raise ValueError(f"unknown metric: {metric}")


# ========================
# diagnostics
# ========================
def save_margin_quantiles(oof: pd.DataFrame, outdir: Path) -> None:
    oof = oof.copy()
    oof["margin"] = oof["max_sim"] - oof["second_sim"]

    qs = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    summary = pd.DataFrame({
        "quantile": qs,
        "max_sim":   [oof["max_sim"].quantile(q) for q in qs],
        "second_sim":[oof["second_sim"].quantile(q) for q in qs],
        "margin":    [oof["margin"].quantile(q) for q in qs],
    })
    summary.to_csv(outdir / "margin_quantiles.csv", index=False)

    # 追加の簡易統計も保存
    stats = {
        "n": int(len(oof)),
        "margin_lt0_ratio": float((oof["margin"] < 0).mean()),
        "margin_eq0_ratio": float((oof["margin"] == 0).mean()),
    }
    for t in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        stats[f"margin_lt_{t}"] = float((oof["margin"] < t).mean())

    pd.DataFrame([stats]).to_csv(outdir / "margin_stats.csv", index=False)

    print("OOF size:", len(oof))
    print(summary.to_string(index=False))
    print("\nmargin < 0 ratio:", stats["margin_lt0_ratio"])
    print("margin == 0 ratio:", stats["margin_eq0_ratio"])
    for t in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        print(f"margin < {t:>5}: {stats[f'margin_lt_{t}']:.4f}")


# ========================
# main
# ========================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # load config block
    infer_cfg = OmegaConf.to_container(cfg["200_optimize_threshold"], resolve=True)
    config = dict_to_namespace(infer_cfg)

    # debug exp name
    if getattr(config, "debug", False):
        config.exp = "200_optimize_threshold_debug"

    # outputs (exp配下に全部入れる)
    exp_dir = Path(config.train_output_dir) / str(config.exp)
    savedir = exp_dir / "threshold"
    savedir.mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, savedir / "200_config.yaml")

    # load oof
    oof_csv = exp_dir / "oof" / "oof_df.csv"
    if not oof_csv.exists():
        raise FileNotFoundError(f"oof_df.csv not found: {oof_csv}")

    oof = pd.read_csv(oof_csv)
    need = {"y", "pred", "max_sim", "second_sim"}
    if not need.issubset(set(oof.columns)):
        raise ValueError(f"oof_df.csv must contain {need}. got {list(oof.columns)}")

    # diagnostics
    save_margin_quantiles(oof, savedir)

    y_true = oof["y"].to_numpy(dtype=int)
    pred_class = oof["pred"].to_numpy(dtype=int)
    max_sim = oof["max_sim"].to_numpy(dtype=float)
    second_sim = oof["second_sim"].to_numpy(dtype=float)

    metric = str(getattr(config, "metric", "macro_f1"))
    unknown_mode = str(getattr(config, "unknown_mode", "holdout_ids"))
    holdout_list = list(getattr(config, "holdout_ids", [])) if unknown_mode == "holdout_ids" else []

    # search space
    thrs = np.arange(float(config.thr_min), float(config.thr_max) + 1e-9, float(config.thr_step))
    mthrs = np.arange(float(config.mthr_min), float(config.mthr_max) + 1e-9, float(config.mthr_step))

    combine_mode = str(getattr(config, "combine_mode", "or"))  # 推奨: or
    save_grid = bool(getattr(config, "save_grid", False))

    # baseline（unknown無しのスコアも残す：CVメモ用）
    base_score = float(f1_score(y_true, pred_class, average="macro"))
    with open(savedir / "baseline.txt", "w") as f:
        f.write(f"macro_f1_no_unknown={base_score}\n")
    print("\n[baseline] macro_f1 (no unknown):", base_score)

    if len(holdout_list) == 0:
        print("\n[WARN] holdout_ids is empty. Nothing to optimize.")
        return

    records = []
    for hid in holdout_list:
        hid = int(hid)
        best_score = -1e9
        best_thr = None
        best_mthr = None

        grid_records = []  # (holdout_id, thr, mthr, score)

        for thr_sim in thrs:
            for thr_margin in mthrs:
                s = score_with_unknown_2d(
                    y_true=y_true,
                    pred_class=pred_class,
                    max_sim=max_sim,
                    second_sim=second_sim,
                    thr_sim=float(thr_sim),
                    thr_margin=float(thr_margin),
                    unknown_ids=[hid],
                    metric=metric,
                    combine_mode=combine_mode,
                )

                if save_grid:
                    grid_records.append((hid, float(thr_sim), float(thr_margin), float(s)))

                if s > best_score:
                    best_score = float(s)
                    best_thr = float(thr_sim)
                    best_mthr = float(thr_margin)

        records.append({
            "holdout_id": hid,
            "best_score": best_score,
            "best_thr": best_thr,
            "best_mthr": best_mthr,
        })

        if save_grid:
            grid_df = pd.DataFrame(grid_records, columns=["holdout_id", "thr", "mthr", "score"])
            grid_df.to_csv(savedir / f"thr2d_grid_id{hid}.csv", index=False)

    df_runs = pd.DataFrame(records).sort_values("holdout_id")
    df_runs.to_csv(savedir / "thr2d_runs.csv", index=False)

    thr_median = float(np.median(df_runs["best_thr"].values))
    mthr_median = float(np.median(df_runs["best_mthr"].values))

    # thr==0 の変なケースを避けたいなら nonzero も出す
    nz = df_runs[df_runs["best_thr"] > 0.0]
    thr_median_nz = float(np.median(nz["best_thr"].values)) if len(nz) else thr_median
    mthr_median_nz = float(np.median(nz["best_mthr"].values)) if len(nz) else mthr_median

    # 代表CVスコアを「median閾値」で計算して保存（これが“CVはどれ？”の答え）
    # holdout_ids を全部 unknown 扱いにする（=疑似unknown混在評価）
    cv_score_median = score_with_unknown_2d(
        y_true=y_true,
        pred_class=pred_class,
        max_sim=max_sim,
        second_sim=second_sim,
        thr_sim=thr_median,
        thr_margin=mthr_median,
        unknown_ids=[int(x) for x in holdout_list],
        metric=metric,
        combine_mode=combine_mode,
    )
    cv_score_median_nz = score_with_unknown_2d(
        y_true=y_true,
        pred_class=pred_class,
        max_sim=max_sim,
        second_sim=second_sim,
        thr_sim=thr_median_nz,
        thr_margin=mthr_median_nz,
        unknown_ids=[int(x) for x in holdout_list],
        metric=metric,
        combine_mode=combine_mode,
    )

    with open(savedir / "best_2d.txt", "w") as f:
        f.write(f"thr_median={thr_median}\n")
        f.write(f"mthr_median={mthr_median}\n")
        f.write(f"thr_median_nonzero={thr_median_nz}\n")
        f.write(f"mthr_median_nonzero={mthr_median_nz}\n")
        f.write(f"metric={metric}\n")
        f.write(f"combine_mode={combine_mode}\n")
        f.write(f"holdout_ids={holdout_list}\n")
        f.write(f"baseline_macro_f1_no_unknown={base_score}\n")
        f.write(f"cv_score_median={cv_score_median}\n")
        f.write(f"cv_score_median_nonzero={cv_score_median_nz}\n")

    print("\n", df_runs)
    print("thr_median:", thr_median, "mthr_median:", mthr_median)
    print("thr_median_nonzero:", thr_median_nz, "mthr_median_nonzero:", mthr_median_nz)
    print(f"[cv] ({combine_mode}) score @ median thr/mthr:", cv_score_median)
    print(f"[cv] ({combine_mode}) score @ median_nonzero thr/mthr:", cv_score_median_nz)
    print("saved:", savedir)


if __name__ == "__main__":
    main()