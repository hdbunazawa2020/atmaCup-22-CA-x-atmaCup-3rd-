from pathlib import Path
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from .transforms import get_train_transform, get_val_transform
from .player_dataset import PlayerDataset


class PlayerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_pp_csv: Path,
        img_size: int,
        batch_size: int,
        num_workers: int,
        num_features: list[str],
        split_method: str = "holdout",         # holdout / sgkf
        val_quarter_from: str = "Q2-016",      # holdout用
        n_splits: int = 5,                     # sgkf用
        fold: int = 0,                         # sgkf用
        seed: int = 42,
        p_hflip: float = 0.5,
    ):
        super().__init__()
        self.train_pp_csv = Path(train_pp_csv)
        self.img_size = int(img_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.num_features = list(num_features)

        self.split_method = str(split_method)
        self.val_quarter_from = str(val_quarter_from)
        self.n_splits = int(n_splits)
        self.fold = int(fold)
        self.seed = int(seed)
        self.p_hflip = float(p_hflip)

        self.train_df = None
        self.val_df = None

    def setup(self, stage: str | None = None):
        df = pd.read_csv(self.train_pp_csv)

        # angle_id を作る（top=1, side=0）
        df["angle_id"] = (df["angle"] == "top").astype(np.float32)

        # frame内人数（rank_x反転などに使えるし、デバッグにも便利）
        n_players = df.groupby(["quarter", "session", "frame"]).size().rename("n_players")
        df = df.merge(n_players, on=["quarter", "session", "frame"], how="left")
        df["n_players"] = df["n_players"].fillna(10).astype(int)

        # NaN埋め（特に mean_dist 等）
        for c in self.num_features:
            if c not in df.columns:
                raise KeyError(f"num_feature not found in train pp csv: {c}")
            df[c] = df[c].fillna(0.0)

        # split
        if self.split_method == "holdout":
            val_mask = df["quarter"] >= self.val_quarter_from
            self.train_df = df[~val_mask].reset_index(drop=True)
            self.val_df = df[val_mask].reset_index(drop=True)
        elif self.split_method == "sgkf":
            sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            groups = df["quarter"].astype(str)  # まずはquarter単位（必要なら session/frame 単位も可）
            y = df["label_id"].astype(int)
            tr_idx, va_idx = None, None
            for f, (tr, va) in enumerate(sgkf.split(df, y=y, groups=groups)):
                if f == self.fold:
                    tr_idx, va_idx = tr, va
                    break
            self.train_df = df.iloc[tr_idx].reset_index(drop=True)
            self.val_df = df.iloc[va_idx].reset_index(drop=True)
        else:
            raise ValueError(f"unknown split_method: {self.split_method}")

        print(f"[DataModule] train={len(self.train_df)} val={len(self.val_df)}")
        print(f"[DataModule] val quarters: {sorted(self.val_df['quarter'].unique().tolist())}")

    def train_dataloader(self):
        ds = PlayerDataset(
            df=self.train_df,
            transform=get_train_transform(self.img_size),
            num_features=self.num_features,
            is_train=True,
            p_hflip=self.p_hflip,
            cache_images=False,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        ds = PlayerDataset(
            df=self.val_df,
            transform=get_val_transform(self.img_size),
            num_features=self.num_features,
            is_train=False,
            p_hflip=0.0,
            cache_images=False,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )