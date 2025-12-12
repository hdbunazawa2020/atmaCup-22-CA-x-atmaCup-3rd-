import os
import random
import pickle
import types
import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Union, Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.distributed as dist
import yaml
from IPython.display import display


# ========== 汎用表示・ログ系 ==========
def sep(word: str, num: int = 80):
    print("=" * num)
    print(word)
    print("=" * num)

def show_df(df: Union[pd.DataFrame, pl.DataFrame], num: int = 3, show_tail: bool = False):
    """データフレームの概要を表示"""
    print(df.shape)
    display(df.head(num))
    if show_tail:
        display(df.tail(num))


# ========== ファイル操作 ==========
def glob_walk(root: Union[str, Path], pattern: str) -> list[Path]:
    """指定パターンにマッチするファイルパスを取得"""
    path = Path(root)
    return sorted(list(path.glob(pattern)))

def to_pickle_file(filename: Union[str, Path], obj: Any):
    """Pickleで保存"""
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle_file(filename: Union[str, Path]) -> Any:
    """Pickleで読み込み"""
    with open(filename, "rb") as f:
        return pickle.load(f)


# ========== データ前処理 ==========
def check_valid_columns(data_paths: list[Path]) -> list[str]:
    """全Nullではないカラムの抽出"""
    df = pl.read_parquet(data_paths[0])
    valid_cols = []
    for col in df.columns:
        if (df[col].null_count() != len(df)) and (df[col].min() != df[col].max()):
            valid_cols.append(col)
    valid_cols += ["vehicle_model"]
    print(f"Valid columns: {valid_cols}")
    return valid_cols

def normalize_tensor(y: torch.Tensor) -> torch.Tensor:
    """2次元Tensor (N, 2) を各列ごとに標準化"""
    for i in range(y.shape[1]):
        mean = y[:, i].mean().item()
        std = y[:, i].std().item()
        y[:, i] = (y[:, i] - mean) / (std + 1e-16)
    return y

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """メモリ節約のための型変換"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage before optimization: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if np.iinfo(np.int8).min < c_min < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif np.iinfo(np.int16).min < c_min < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif np.iinfo(np.int32).min < c_min < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:  # float
                if np.finfo(np.float16).min < c_min < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif np.finfo(np.float32).min < c_min < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization: {end_mem:.2f} MB (Reduced {100*(start_mem-end_mem)/start_mem:.1f}%)")
    return df


# ========== シード固定 ==========
def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ========== 分散学習系 ==========
def init_distributed(rank: int, world_size: int):
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        print("Single-GPU mode (no distributed setup)")

def cleanup_distributed():
    dist.barrier()
    dist.destroy_process_group()


# ========== 計測・ログ ==========
def format_elapsed_time(elapsed: float) -> str:
    return str(datetime.timedelta(seconds=int(round(elapsed))))


# ========== Config保存系 ==========
def namespace_to_dict(obj: Any) -> dict:
    """SimpleNamespaceやネストされた構造を辞書に再帰変換"""
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(i) for i in obj]
    else:
        return obj

def save_config_yaml(config_obj: Any, save_path: Union[str, Path]):
    config_dict = namespace_to_dict(config_obj)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

def dict_to_namespace(d: Any) -> Any:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d