#!/usr/bin/env python
"""
datamodule_2.py

Build PyTorch DataLoader objects for the representation-learning model.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class SepticDataset(Dataset):
    def __init__(self, path: Path):
        d = torch.load(path)
        self.x_static = torch.from_numpy(d["x_static"]).float()
        self.x_seq = torch.from_numpy(d["x_seq"]).float()
        self.mask = torch.from_numpy(d["mask"]).float()
        self.stay_id = d.get("stay_id")

    def __len__(self) -> int:
        return self.x_static.shape[0]

    def __getitem__(self, idx):
        return {
            "seq": torch.nan_to_num(self.x_seq[idx], nan=0.0),
            "mask": self.mask[idx],
            "static": self.x_static[idx],
        }


def make_loaders(
    batch: int = 128,
    num_workers: int = 0,
    data_dir: str = "septic_shock_peer_review/results/data_proc",
):
    data_dir_path = Path(data_dir)

    def _loader(split: str):
        ds = SepticDataset(data_dir_path / f"{split}_tensor.pt")
        return DataLoader(
            ds,
            batch_size=batch,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return {split: _loader(split) for split in ("train", "val", "test")}
