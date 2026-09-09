"""Data loading and small training utilities."""

import io
import os
import random
from typing import Tuple

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


class ParquetImageDataset(Dataset):
    """CIFAR-10/100 loaded from a HuggingFace-hosted parquet file (column
    'img': {'bytes': <PNG>, 'path': ...}, plus a label column) -- a much
    faster download path than torchvision's default (throttled) source for
    this environment. Decodes PNG bytes to PIL images lazily in
    __getitem__. label_column differs by dataset: CIFAR-10's mirror uses
    'label'; CIFAR-100's uses 'fine_label' (100 classes) with a separate
    'coarse_label' (20 superclasses) that we don't use here."""

    def __init__(self, parquet_path: str, transform=None, label_column: str = "label"):
        import pandas as pd
        from PIL import Image
        self._Image = Image
        df = pd.read_parquet(parquet_path, columns=["img", label_column])
        self.img_bytes = [row["bytes"] for row in df["img"]]
        self.labels = df[label_column].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self._Image.open(io.BytesIO(self.img_bytes[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CIFAR_MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
CIFAR_STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}


def get_cifar_loaders(dataset: str, data_dir: str, batch_size: int,
                       train_subset: int = None, val_subset: int = None,
                       num_workers: int = 4, hf_parquet_dir: str = None) -> Tuple[DataLoader, DataLoader]:
    """If hf_parquet_dir is given (containing train-*.parquet / test-*.parquet
    from a HuggingFace 'plain_text' CIFAR mirror), load from there instead of
    torchvision's default download source, which was observed to be heavily
    throttled (~110 KB/s) in this environment."""
    assert dataset in ("cifar10", "cifar100")
    mean, std = CIFAR_MEAN[dataset], CIFAR_STD[dataset]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if hf_parquet_dir is not None:
        train_path = os.path.join(hf_parquet_dir, "train-00000-of-00001.parquet")
        test_path = os.path.join(hf_parquet_dir, "test-00000-of-00001.parquet")
        label_col = "label" if dataset == "cifar10" else "fine_label"
        train_data = ParquetImageDataset(train_path, transform=train_tf, label_column=label_col)
        val_data = ParquetImageDataset(test_path, transform=val_tf, label_column=label_col)
    else:
        cls = dset.CIFAR10 if dataset == "cifar10" else dset.CIFAR100
        train_data = cls(root=data_dir, train=True, download=True, transform=train_tf)
        val_data = cls(root=data_dir, train=False, download=True, transform=val_tf)

    if train_subset is not None:
        train_data = Subset(train_data, list(range(min(train_subset, len(train_data)))))
    if val_subset is not None:
        val_data = Subset(val_data, list(range(min(val_subset, len(val_data)))))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item() * 100.0


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch
