import os, gc, re, yaml, glob, pickle, warnings
import time
import random, math
import joblib, pickle, itertools
from pathlib import Path

import numpy as np
import scipy as sp
import polars as pl
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import warnings
# warnings.filterwarnings('ignore')

# original
import sys
sys.path.append(r"..")
import utils
from utils.data import sep, show_df, glob_walk, set_seed, format_time, save_config_yaml, dict_to_namespace
from utils.wandb_utils import set_wandb, safe_wandb_config

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")

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
    config_dict = OmegaConf.to_container(cfg["xxx_filename"], resolve=True)
    config = dict_to_namespace(config_dict)
    # when debug
    if config.debug:
        config.exp = "xxx_debug" # TODO: ファイルの連番を入れる
    # set WandB
    if config.use_wandb:
        set_wandb(config)
    # make savedir
    savedir = Path(config.output_dir) / config.exp
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir / "oof", exist_ok=True)
    os.makedirs(savedir / "yaml", exist_ok=True)
    os.makedirs(savedir / "model", exist_ok=True)

    # YAMLとして保存
    output_path = Path(savedir/"yaml"/"config.yaml")
    save_config_yaml(config, output_path)
    print(f"Config saved to {output_path.resolve()}")

    # ==================
    # main
    # ==================
    


if __name__ == "__main__":
    main()
