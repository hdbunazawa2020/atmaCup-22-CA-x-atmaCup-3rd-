import random
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _flip_tabular_inplace(x: torch.Tensor, col2idx: dict[str, int], n_players: int):
    """
    flipに合わせて数値特徴を補正（存在するものだけ）
    - fx: 符号反転
    - rank_x: (n-1 - rank_x)
    - angle_id はそのまま（top/sideの属性なので）
    """
    if "fx" in col2idx:
        x[col2idx["fx"]] = -x[col2idx["fx"]]

    if "rank_x" in col2idx:
        x[col2idx["rank_x"]] = (float(n_players) - 1.0) - x[col2idx["rank_x"]]

    return x


class PlayerDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        num_features: Sequence[str],
        is_train: bool,
        p_hflip: float = 0.5,
        cache_images: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.num_features = list(num_features)
        self.is_train = bool(is_train)
        self.p_hflip = float(p_hflip)
        self.cache_images = bool(cache_images)
        self._cache = {}

        self.col2idx = {c: i for i, c in enumerate(self.num_features)}

        # 欠損は0で埋める（Lightning側でもOKだがここで安全に）
        for c in self.num_features:
            if c not in self.df.columns:
                raise KeyError(f"num_feature column not found: {c}")
            self.df[c] = self.df[c].fillna(0.0)

        # n_players は flip rank_x に必要（無ければ 10 で固定）
        if "n_players" not in self.df.columns:
            self.df["n_players"] = 10

    def __len__(self):
        return len(self.df)

    def _load_rgb(self, path: str) -> np.ndarray:
        if self.cache_images and path in self._cache:
            return self._cache[path]

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"image not found or unreadable: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.cache_images:
            self._cache[path] = img
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        crop_path = str(row["crop_path"])
        img = self._load_rgb(crop_path)

        x_num = torch.tensor(row[self.num_features].to_numpy(np.float32), dtype=torch.float32)
        n_players = int(row["n_players"])

        # ---- horizontal flip (manual) ----
        if self.is_train and random.random() < self.p_hflip:
            img = cv2.flip(img, 1)  # horizontal
            x_num = _flip_tabular_inplace(x_num, self.col2idx, n_players)

        transformed = self.transform(image=img)
        image = transformed["image"]  # torch tensor (C,H,W)

        if "label_id" in row.index:
            y = torch.tensor(int(row["label_id"]), dtype=torch.long)
            return {"image": image, "num": x_num, "label": y, "n_players": n_players}
        else:
            return {"image": image, "num": x_num, "n_players": n_players}