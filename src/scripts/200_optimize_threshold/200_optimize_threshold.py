import os
from pathlib import Path
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, accuracy_score

# original
import sys
sys.path.append(r"..")
import utils
from utils.data import save_config_yaml, dict_to_namespace

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")

# ========================
# utils
# ========================
def score_with_unknown(y_true, pred_class, max_sim, thr, unknown_ids=None, metric="macro_f1"):
    """
    pred_class: valでの argmax 予測（0..K-1）
    max_sim:    valでの最大類似度
    thr:        unknown閾値。max_sim < thr は -1 扱い
    unknown_ids: y_trueのうち、指定IDを疑似unknown(-1)扱いにする
    """
    y = y_true.copy()
    if unknown_ids is not None and len(unknown_ids) > 0:
        y = np.where(np.isin(y, unknown_ids), -1, y)

    pred = pred_class.copy()
    pred = np.where(max_sim < thr, -1, pred)

    # -1 を含めて評価（クラスに -1 を含める）
    labels = np.unique(np.concatenate([y, pred]))
    labels = labels.tolist()

    if metric == "macro_f1":
        return f1_score(y, pred, average="macro", labels=labels)
    elif metric == "accuracy":
        return accuracy_score(y, pred)
    else:
        raise ValueError(metric)


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
    config_dict = OmegaConf.to_container(cfg["200_optimize_threshold"], resolve=True)
    config = dict_to_namespace(config_dict)
    # when debug
    if config.debug:
        config.exp = "200_optimize_threshold_debug" # TODO: ファイルの連番を入れる
    savedir = Path(config.train_output_dir) / config.train_exp / "threshold"
    savedir.mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, savedir / "config.yaml")

    # OOF: y, pred, max_sim
    config.oof_csv = Path(config.train_output_dir) / config.train_exp / "oof" / "oof_df.csv"
    oof = pd.read_csv(config.oof_csv)
    if not {"y", "pred", "max_sim"}.issubset(set(oof.columns)):
        raise ValueError(f"oof_df.csv must contain y,pred,max_sim. got {oof.columns}")

    y_true = oof["y"].to_numpy().astype(int)
    pred_class = oof["pred"].to_numpy().astype(int)
    max_sim = oof["max_sim"].to_numpy().astype(float)

    # unknown ids
    if config.unknown_mode == "holdout_ids":
        unknown_ids = list(config.holdout_ids)
    else:
        unknown_ids = []
    # unknown ids list
    if config.unknown_mode == "holdout_ids":
        holdout_list = list(config.holdout_ids)
    else:
        holdout_list = []
    thrs = np.arange(config.thr_min, config.thr_max + 1e-9, config.thr_step)

    # --- run per holdout id ---
    records = []
    if len(holdout_list) == 0:
        # no unknown simulation
        scores = []
        for thr in thrs:
            s = score_with_unknown(
                y_true=y_true,
                pred_class=pred_class,
                max_sim=max_sim,
                thr=float(thr),
                unknown_ids=[],
                metric=config.metric,
            )
            scores.append(s)
        scores = np.array(scores)
        best_i = int(scores.argmax())
        records.append({
            "holdout_id": None,
            "best_thr": float(thrs[best_i]),
            "best_score": float(scores[best_i]),
        })
    else:
        for hid in holdout_list:
            scores = []
            for thr in thrs:
                s = score_with_unknown(
                    y_true=y_true,
                    pred_class=pred_class,
                    max_sim=max_sim,
                    thr=float(thr),
                    unknown_ids=[int(hid)],
                    metric=config.metric,
                )
                scores.append(s)
            scores = np.array(scores)
            best_i = int(scores.argmax())
            records.append({
                "holdout_id": int(hid),
                "best_thr": float(thrs[best_i]),
                "best_score": float(scores[best_i]),
            })

    df_runs = pd.DataFrame(records).sort_values(["holdout_id"], na_position="first")
    df_runs.to_csv(savedir / "thr_runs.csv", index=False)

    # --- summary ---
    thrs_best = df_runs["best_thr"].to_numpy(np.float32)
    scores_best = df_runs["best_score"].to_numpy(np.float32)

    thr_mean = float(np.mean(thrs_best))
    thr_median = float(np.median(thrs_best))
    thr_std = float(np.std(thrs_best))

    score_mean = float(np.mean(scores_best))
    score_median = float(np.median(scores_best))

    with open(savedir / "summary.txt", "w") as f:
        f.write(f"metric={config.metric}\n")
        f.write(f"holdout_ids={holdout_list}\n")
        f.write(f"thr_mean={thr_mean}\n")
        f.write(f"thr_median={thr_median}\n")
        f.write(f"thr_std={thr_std}\n")
        f.write(f"best_score_mean={score_mean}\n")
        f.write(f"best_score_median={score_median}\n")

    print("saved:", savedir)
    print(df_runs)
    print(f"[thr] mean={thr_mean:.3f} median={thr_median:.3f} std={thr_std:.3f}")
    print(f"[score] mean={score_mean:.3f} median={score_median:.3f}")

    th = df_runs["best_thr"].to_numpy()
    print("median_nonzero:", float(np.median(th[th > 0])))
    print("mean_nonzero:", float(np.mean(th[th > 0])))

if __name__ == "__main__":
    main()