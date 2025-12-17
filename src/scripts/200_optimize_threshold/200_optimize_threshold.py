import os
from pathlib import Path
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
# utils
# ========================
def score_with_unknown_2d(
    y_true, pred_class, max_sim, second_sim,
    thr_sim, thr_margin,
    unknown_ids=None, metric="macro_f1",
    combine_mode: str = "and",   # "and" or "or"
):
    y = y_true.copy()
    if unknown_ids is not None and len(unknown_ids) > 0:
        y = np.where(np.isin(y, unknown_ids), -1, y)

    margin = max_sim - second_sim
    pred = pred_class.copy()

    cond_sim = (max_sim < thr_sim)
    cond_m   = (margin < thr_margin)

    if combine_mode == "and":
        unknown_cond = cond_sim & cond_m
    elif combine_mode == "or":
        unknown_cond = cond_sim | cond_m
    else:
        raise ValueError(f"unknown combine_mode: {combine_mode}")

    pred = np.where(unknown_cond, -1, pred)

    labels = np.unique(np.concatenate([y, pred])).tolist()
    if metric == "macro_f1":
        return f1_score(y, pred, average="macro", labels=labels)
    elif metric == "accuracy":
        return accuracy_score(y, pred)
    else:
        raise ValueError(metric)


# ===================================
# main
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg["200_optimize_threshold"], resolve=True)
    config = dict_to_namespace(config_dict)

    if config.debug:
        config.exp = "200_optimize_threshold_debug"

    exp_dir = Path(config.train_output_dir) / config.exp
    savedir = exp_dir / "threshold"
    savedir.mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, savedir / "200_config.yaml")

    # OOF
    oof_csv = exp_dir / "oof" / "oof_df.csv"
    if not oof_csv.exists():
        raise FileNotFoundError(f"oof_df.csv not found: {oof_csv}")
    oof = pd.read_csv(oof_csv)

    need = {"y", "pred", "max_sim", "second_sim"}
    if not need.issubset(set(oof.columns)):
        raise ValueError(f"oof_df.csv must contain {need}. got {oof.columns}")

    oof["margin"] = oof["max_sim"].astype(float) - oof["second_sim"].astype(float)

    # 分布チェック（根拠ありで grid を決める材料）
    qs = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    summary = pd.DataFrame({
        "max_sim": oof["max_sim"].quantile(qs),
        "second_sim": oof["second_sim"].quantile(qs),
        "margin": oof["margin"].quantile(qs),
    }).rename_axis("quantile").reset_index()
    print("OOF size:", len(oof))
    print(summary.to_string(index=False))
    print("\nmargin < 0 ratio:", float((oof["margin"] < 0).mean()))
    for t in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        print(f"margin < {t:>5}: {(oof['margin'] < t).mean():.4f}")

    y_true = oof["y"].to_numpy().astype(int)
    pred_class = oof["pred"].to_numpy().astype(int)
    max_sim = oof["max_sim"].to_numpy().astype(float)
    second_sim = oof["second_sim"].to_numpy().astype(float)

    combine_mode = str(getattr(config, "combine_mode", "and"))
    print("\ncombine_mode:", combine_mode)

    holdout_list = list(config.holdout_ids) if getattr(config, "unknown_mode", "holdout_ids") == "holdout_ids" else []
    if not holdout_list:
        raise ValueError("holdout_ids is empty. set unknown_mode=holdout_ids and holdout_ids=[...]")

    thrs = np.arange(config.thr_min, config.thr_max + 1e-9, config.thr_step)
    mthrs = np.arange(config.mthr_min, config.mthr_max + 1e-9, config.mthr_step)

    records = []
    save_grid = bool(getattr(config, "save_grid", False))

    for hid in holdout_list:
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
                    unknown_ids=[int(hid)],
                    metric=config.metric,
                    combine_mode=combine_mode,
                )

                if save_grid:
                    grid_records.append((int(hid), float(thr_sim), float(thr_margin), float(s)))

                if s > best_score:
                    best_score = float(s)
                    best_thr = float(thr_sim)
                    best_mthr = float(thr_margin)

        records.append({
            "holdout_id": int(hid),
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

    nz = df_runs[df_runs["best_thr"] > 0.0]
    thr_median_nz = float(np.median(nz["best_thr"].values)) if len(nz) else thr_median
    mthr_median_nz = float(np.median(nz["best_mthr"].values)) if len(nz) else mthr_median

    with open(savedir / "best_2d.txt", "w") as f:
        f.write(f"thr_median={thr_median}\n")
        f.write(f"mthr_median={mthr_median}\n")
        f.write(f"thr_median_nonzero={thr_median_nz}\n")
        f.write(f"mthr_median_nonzero={mthr_median_nz}\n")
        f.write(f"metric={config.metric}\n")
        f.write(f"combine_mode={combine_mode}\n")
        f.write(f"holdout_ids={holdout_list}\n")

    print(df_runs)
    print("thr_median:", thr_median, "mthr_median:", mthr_median)
    print("thr_median_nonzero:", thr_median_nz, "mthr_median_nonzero:", mthr_median_nz)
    print("saved:", savedir)


if __name__ == "__main__":
    main()