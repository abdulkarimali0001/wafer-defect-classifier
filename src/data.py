"""Load the MixedWM38 wafer-map dataset and prepare it for training.

Each wafer map is a 52x52 grid of dies:
    0 = no die (outside the wafer)
    1 = die that passed the electrical test
    2 = die that failed the electrical test

Each map has 8 binary labels (multi-label): one wafer can show several
defect patterns at once, e.g. Center + Scratch. A map with all zeros is Normal.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Label order of arr_1, the 8 basic defect types (C2-C9 in the dataset paper).
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_Full", "Scratch", "Random"]


def load_npz(path: str | Path):
    """Return (maps, labels): maps uint8 (N, 52, 52), labels float32 (N, 8)."""
    d = np.load(path)
    maps = d["arr_0"].astype(np.uint8)
    # The README documents values 0/1/2, but 214 cells (in 105 of 38,015 maps) hold 3.
    # They are undocumented, so we treat them as failed dies (2).
    maps = np.minimum(maps, 2)
    labels = d["arr_1"].astype(np.float32)
    return maps, labels


def one_hot_maps(maps: np.ndarray) -> np.ndarray:
    """Turn values {0,1,2} into 3 channels: (no die, pass, fail). Shape (N, 3, 52, 52)."""
    return np.stack([(maps == v) for v in (0, 1, 2)], axis=1).astype(np.float32)


def stratified_split(labels: np.ndarray, val=0.15, test=0.15, seed=42):
    """Split indices so every label combination (all 38 patterns) appears in each split."""
    rng = np.random.default_rng(seed)
    combo = np.array(["".join(map(str, row.astype(int))) for row in labels])
    tr, va, te = [], [], []
    for c in np.unique(combo):
        idx = rng.permutation(np.where(combo == c)[0])
        n_te, n_va = int(len(idx) * test), int(len(idx) * val)
        te += list(idx[:n_te]); va += list(idx[n_te:n_te + n_va]); tr += list(idx[n_te + n_va:])
    return np.array(tr), np.array(va), np.array(te)


class WaferDataset(Dataset):
    """Wafer maps as tensors. With augment=True, applies random flips and 90-degree
    rotations: a defect pattern keeps its type when the wafer is rotated or mirrored."""

    def __init__(self, maps: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.x = torch.from_numpy(one_hot_maps(maps))
        self.y = torch.from_numpy(labels)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.augment:
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(1, 2))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(2,))
        return x, self.y[i]


def pattern_name(label_row) -> str:
    """'Center+Scratch' style name for a label vector; 'Normal' if no defect."""
    names = [c for c, v in zip(CLASSES, label_row) if v >= 0.5]
    return "+".join(names) if names else "Normal"
