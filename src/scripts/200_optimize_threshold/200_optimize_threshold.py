import os
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, accuracy_score

import sys
sys.path.append(r"..")
from utils.data import save_config_yaml, dict_to_namespace

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")

# ========================
# scoring
# ========================
def _labels_with_unknown(y: np.ndarray, pred: np.ndarray) -> list[int]:
    return np.unique(np.concatenate([y, pred])).tolist()

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
    y = y_true.copy()
    if unknown_ids:
        y = np.where(np.isin(y, unknown_ids), -1, y)

    margin = max_sim - second_sim
    pred = pred_class.copy()

    cond_sim = (max_sim < thr_sim)
    cond_m = (margin < thr_margin)

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

def save_margin_quantiles(oof: pd.DataFrame, outdir: Path) -> None:
    oof = oof.copy()
    oof["margin"] = oof["max_sim"].astype(float) - oof["second_sim"].astype(float)

    qs = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    summary = pd.DataFrame({
        "quantile": qs,
        "max_sim":   [oof["max_sim"].quantile(q) for q in qs],
        "second_sim":[oof["second_sim"].quantile(q) for q in qs],
        "margin":    [oof["margin"].quantile(q) for q in qs],
    })
    summary.to_csv(outdir / "margin_quantiles.csv", index=False)

    stats = {
        "n": int(len(oof)),
        "margin_lt0_ratio": float((oof["margin"] < 0).mean()),
        "margin_eq0_ratio": float((oof["margin"] == 0).mean()),
    }
    for t in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        stats[f"margin_lt_{t}"] = float((oof["margin"] < t).mean())

    pd.DataFrame([stats]).to_csv(outdir / "margin_stats.csv", index=False)

    print("OOF size:", len(oof))
    print(summary.drop(columns=["quantile"]).rename_axis("quantile").reset_index().to_string(index=False))
    print("\nmargin < 0 ratio:", stats["margin_lt0_ratio"])
    for t in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        print(f"margin < {t:>5}: {stats[f'margin_lt_{t}']:.4f}")

# ========================
# helpers for folds/OOF
# ========================
def _strip_fold_suffix(exp: str) -> str:
    return re.sub(r"_fold\d+$", "", exp)

def find_oof_csv(exp_dir: Path) -> Path:
    oof_dir = exp_dir / "oof"
    p = oof_dir / "oof_df.csv"
    if p.exists():
        return p
    cands = sorted(oof_dir.glob("oof_df*.csv"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        raise FileNotFoundError(f"OOF not found in: {oof_dir} (expected oof_df.csv or oof_df*.csv)")
    raise FileNotFoundError(f"Multiple OOF files found in {oof_dir}: {cands}")

# ========================
# per-exp run
# ========================
def run_one_exp(exp_dir: Path, config) -> None:
    exp_dir = Path(exp_dir)
    print(f"\n==============================")
    print(f"[200] exp_dir = {exp_dir}")
    print(f"==============================")

    savedir = exp_dir / "threshold"
    savedir.mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, savedir / "200_config.yaml")

    # load oof
    oof_csv = find_oof_csv(exp_dir)
    print("[oof] loaded:", oof_csv)

    oof = pd.read_csv(oof_csv)
    need = {"y", "pred", "max_sim", "second_sim"}
    if not need.issubset(set(oof.columns)):
        raise ValueError(f"oof_df.csv must contain {need}. got {list(oof.columns)}")

    save_margin_quantiles(oof, savedir)

    y_true = oof["y"].to_numpy(dtype=int)
    pred_class = oof["pred"].to_numpy(dtype=int)
    max_sim = oof["max_sim"].to_numpy(dtype=float)
    second_sim = oof["second_sim"].to_numpy(dtype=float)

    metric = str(getattr(config, "metric", "macro_f1"))
    combine_mode = str(getattr(config, "combine_mode", "or"))
    print("\ncombine_mode:", combine_mode)

    unknown_mode = str(getattr(config, "unknown_mode", "holdout_ids"))
    holdout_list = list(getattr(config, "holdout_ids", [])) if unknown_mode == "holdout_ids" else []
    if len(holdout_list) == 0:
        raise ValueError("holdout_ids is empty. set unknown_mode=holdout_ids and holdout_ids=[...]")

    thrs = np.arange(float(config.thr_min), float(config.thr_max) + 1e-9, float(config.thr_step))
    mthrs = np.arange(float(config.mthr_min), float(config.mthr_max) + 1e-9, float(config.mthr_step))
    save_grid = bool(getattr(config, "save_grid", False))

    base_score = float(f1_score(y_true, pred_class, average="macro"))
    (savedir / "baseline.txt").write_text(f"macro_f1_no_unknown={base_score}\n")
    print("\n[baseline] macro_f1 (no unknown):", base_score)

    records = []
    for hid in holdout_list:
        hid = int(hid)

        # ★ 保険：そのfoldのvalに存在しないクラスはスキップ
        if (y_true == hid).sum() == 0:
            print(f"[skip] holdout_id={hid} not in y_true. skip.")
            continue

        best_score = -1e9
        best_thr = None
        best_mthr = None
        grid_records = []

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

        records.append({"holdout_id": hid, "best_score": best_score, "best_thr": best_thr, "best_mthr": best_mthr})

        if save_grid:
            pd.DataFrame(grid_records, columns=["holdout_id", "thr", "mthr", "score"]).to_csv(
                savedir / f"thr2d_grid_id{hid}.csv", index=False
            )

    if len(records) == 0:
        raise RuntimeError("No holdout_id was optimized (maybe all holdout ids are absent in this fold).")

    df_runs = pd.DataFrame(records).sort_values("holdout_id")
    df_runs.to_csv(savedir / "thr2d_runs.csv", index=False)

    thr_median = float(np.median(df_runs["best_thr"].values))
    mthr_median = float(np.median(df_runs["best_mthr"].values))
    nz = df_runs[df_runs["best_thr"] > 0.0]
    thr_median_nz = float(np.median(nz["best_thr"].values)) if len(nz) else thr_median
    mthr_median_nz = float(np.median(nz["best_mthr"].values)) if len(nz) else mthr_median

    # cv score (use only ids that were actually optimized)
    used_ids = df_runs["holdout_id"].astype(int).tolist()

    def score_holdout_mean(thr_sim: float, thr_margin: float) -> float:
        ss = []
        for hid in holdout_list:
            ss.append(score_with_unknown_2d(
                y_true=y_true,
                pred_class=pred_class,
                max_sim=max_sim,
                second_sim=second_sim,
                thr_sim=float(thr_sim),
                thr_margin=float(thr_margin),
                unknown_ids=[int(hid)],   # ★ 1つずつ
                metric=metric,
                combine_mode=combine_mode,
            ))
        return float(np.mean(ss))

    cv_score_median = score_holdout_mean(thr_median, mthr_median)
    cv_score_median_nz = score_holdout_mean(thr_median_nz, mthr_median_nz)

    print(f"[cv] holdout-mean score @ median thr/mthr: {cv_score_median}")
    print(f"[cv] holdout-mean score @ median_nonzero thr/mthr: {cv_score_median_nz}")

    with open(savedir / "best_2d.txt", "w") as f:
        f.write(f"thr_median={thr_median}\n")
        f.write(f"mthr_median={mthr_median}\n")
        f.write(f"thr_median_nonzero={thr_median_nz}\n")
        f.write(f"mthr_median_nonzero={mthr_median_nz}\n")
        f.write(f"metric={metric}\n")
        f.write(f"combine_mode={combine_mode}\n")
        f.write(f"holdout_ids={used_ids}\n")
        f.write(f"baseline_macro_f1_no_unknown={base_score}\n")
        f.write(f"cv_score_median={cv_score_median}\n")
        f.write(f"cv_score_median_nonzero={cv_score_median_nz}\n")

    classwise = {str(int(r["holdout_id"])): {
        "thr_sim": float(r["best_thr"]),
        "thr_margin": float(r["best_mthr"]),
        "best_score": float(r["best_score"]),
    } for _, r in df_runs.iterrows()}

    payload = {
        "exp_dir": str(exp_dir),
        "metric": metric,
        "combine_mode": combine_mode,
        "global": {
            "thr_median": thr_median,
            "mthr_median": mthr_median,
            "thr_median_nonzero": thr_median_nz,
            "mthr_median_nonzero": mthr_median_nz,
            "baseline_macro_f1_no_unknown": base_score,
            "cv_score_median": float(cv_score_median),
            "cv_score_median_nonzero": float(cv_score_median_nz),
        },
        "classwise": classwise,
    }

    json_path = savedir / "classwise_thresholds.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n", df_runs)
    print("thr_median:", thr_median, "mthr_median:", mthr_median)
    print("thr_median_nonzero:", thr_median_nz, "mthr_median_nonzero:", mthr_median_nz)
    print(f"[cv] ({combine_mode}) score @ median thr/mthr:", cv_score_median)
    print(f"[cv] ({combine_mode}) score @ median_nonzero thr/mthr:", cv_score_median_nz)
    print("[saved] classwise thresholds:", json_path)
    print("saved:", savedir)

# ========================
# main
# ========================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    infer_cfg = OmegaConf.to_container(cfg["200_optimize_threshold"], resolve=True)
    config = dict_to_namespace(infer_cfg)

    base_exp = _strip_fold_suffix(str(config.exp))
    folds = list(getattr(config, "folds", []))

    if len(folds) > 0:
        exp_dirs = [Path(config.train_output_dir) / f"{base_exp}_fold{int(f)}" for f in folds]
    else:
        exp_dirs = [Path(config.train_output_dir) / str(config.exp)]

    for exp_dir in exp_dirs:
        run_one_exp(exp_dir, config)

if __name__ == "__main__":
    main()