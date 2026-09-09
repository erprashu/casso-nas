"""Data loading and small training utilities."""

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CIFAR_MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
CIFAR_STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}


def get_cifar_loaders(dataset: str, data_dir: str, batch_size: int,
                       train_subset: int = None, val_subset: int = None,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    assert dataset in ("cifar10", "cifar100")
    mean, std = CIFAR_MEAN[dataset], CIFAR_STD[dataset]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

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
