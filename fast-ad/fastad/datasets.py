'''
Fetch, process, and sample data as needed
'''

import os
from typing import Union, Type, Tuple
import urllib
import zipfile
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VisionDataset, MNIST, FashionMNIST, CIFAR10
#from tiny_imagenet_torch import TinyImageNet


class CicadaTransform:
    """ turns (batch, 18, 14) numpy array into (batch, 1, 18, 14) tensor with log scaling """
    def __init__(self, max_value=255):
        self.max_value = max_value
        self.norm_factor = np.log1p(max_value)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.unsqueeze(0)
        return torch.log1p(x) / self.norm_factor

# Seeds chosen deliberately. SEED_80_20 reproduces the original 80/20 split
# so Phase 1 checkpoints from before the three-way split remain valid.
# SEED_50_50 partitions the remaining 20% into val/test independently.
SEED_80_20 = 42
SEED_50_50 = 43

class CICADA(VisionDataset):
    """
    CICADA dataset loader with three-way stratified split.

    Label 0 = Zero Bias (background / inlier)
    Labels 1-10 = Various signal processes (holdouts / anomalies)

    Parameters
    ----------
    split : str
        One of "train" (80%), "val" (10%), or "test" (10%).
        For backward compatibility, pass `train=True` → "train",
        or `train=False` → "val" (matches pre-split behavior).
    """
    def __init__(
        self,
        root: str = "/scratch/network/lo8603/thesis/fast-ad/data/h5_files/",
        transform=None,
        train: bool = None,
        split: str = None,
        **kwargs,
    ):
        self.root = root
        self.transform = transform

        # Resolve split from either `split` or the legacy `train` flag
        if split is None:
            if train is None:
                raise ValueError("Must specify either `split` or `train`")
            split = "train" if train else "val"
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split!r}")
        self.split = split

        self.class_dict = {
            "zb":                0,
            "glugluhtotautau":   1,
            "glugluhtogg":       2,
            "hto2longlivedto4b": 3,
            "singleneutrino":    4,
            "suep":              5,
            "tt":                6,
            "vbfhto2b":          7,
            "vbfhtotautau":      8,
            "zprimetotautau":    9,
            "zz":                10,
        }

        self.data, self.targets = self._load_data()

    def _load_data(self):
        X, y = [], []
        for process, label in self.class_dict.items():
            data_path = os.path.join(self.root, f"{process}.h5")
            if not os.path.exists(data_path):
                print(f"WARNING: {data_path} not found, skipping {process}")
                continue
            with h5py.File(data_path, 'r') as f:
                X.append(f['et_regions'][:])
                y.append(np.full(f['et_regions'].shape[0], label))

        X = np.concatenate(X, axis=0).astype(np.uint8)
        y = np.concatenate(y, axis=0)

        # Step 1: 80/20 stratified split (matches pre-three-way behavior)
        train_size = int(0.8 * len(X))
        X_tr, X_rest, y_tr, y_rest = train_test_split(
            X, y,
            train_size=train_size,
            stratify=y,
            random_state=SEED_80_20,
            shuffle=True,
        )

        if self.split == "train":
            X_out, y_out = X_tr, y_tr
        else:
            # Step 2: split remaining 20% evenly into val/test, stratified again
            X_val, X_te, y_val, y_te = train_test_split(
                X_rest, y_rest,
                test_size=0.5,
                stratify=y_rest,
                random_state=SEED_50_50,
                shuffle=True,
            )
            if self.split == "val":
                X_out, y_out = X_val, y_val
            else:  # test
                X_out, y_out = X_te, y_te

        unique, counts = np.unique(y_out, return_counts=True)
        print(
            f"[CICADA split={self.split}] shape={X_out.shape} "
            f"label distribution: {dict(zip(unique.tolist(), counts.tolist()))}"
        )
        return X_out, y_out

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
class TargetDataset(Dataset):
    def __init__(self, dataset: Dataset, teacher_mapping: callable, distillation_encoder: callable):
        self.dataset = dataset
        self.teacher_mapping = teacher_mapping
        self.distillation_encoder = distillation_encoder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        with torch.no_grad():
            reco_error = self.teacher_mapping(x)
            soft_targets = self.distillation_encoder(reco_error)
        return x, soft_targets


class ExposureDataset(Dataset):
    def __init__(self, size: int = 10000, image_size: tuple = (1, 28, 28)):
        self.size = size
        self.image_size = image_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        raise NotImplemented


class NoiseDataset(ExposureDataset):
    def __getitem__(self, idx):
        # Generate random noise image
        image = torch.randn(self.image_size)
        return image, 42


class OneDataset(ExposureDataset):
    def __getitem__(self, idx):
        return torch.ones(self.image_size), 42


class MonteCarloNegativeDataset(Dataset):
    """
    Oracle negative samples for EBM training: real ZB events only.

    Loads et_regions from zb.h5 and applies CicadaTransform.
    """
    def __init__(self, root: str, transform=None, max_per_file: int = None):
        self.transform = transform or CicadaTransform()
        sl = slice(None, max_per_file)
        path = os.path.join(root, "zb.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"ZB background file not found: {path}")
        with h5py.File(path, "r") as f:
            self.data = f["et_regions"][sl].astype(np.uint8)
        print(f"MonteCarloNegativeDataset: {len(self.data)} ZB events from {root}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, -1  # label -1: background, never treated as anomaly


def get_mc_negative_loader(
    root: str,
    batch_size: int = 512,
    max_per_file: int = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Return an infinite-cycling DataLoader of real ZB events as oracle negative
    samples for EBM training.
    """
    ds = MonteCarloNegativeDataset(root=root, max_per_file=max_per_file)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )


def get_inlier_inidices(targets, hold_out_set: set):
    if isinstance(targets, list):
        targets = np.array(targets)
    elif isinstance(targets, torch.Tensor):
        targets = targets.numpy()
        
    # Store indices of inliers, not just boolean mask
    outlier_mask = np.zeros(len(targets), dtype=bool)

    for hold_out_class in hold_out_set:
        outlier_mask |= (targets == hold_out_class)
    
    return np.where(~outlier_mask)[0]


def get_base_datasets(
    base_name: str,
    root: str = './data',
) -> Tuple[Dataset, Dataset]:
    """
    Get the base dataset
    :param base_name: Name of the dataset. One of ['MNIST', 'FMNIST', 'CIFAR10', 'TinyImageNet']
    :param root: Where to look for or store the dataset
    :return: (train_ds, val_ds): tuple pytorch datasets with binary labels (0: inlier, 1: outlier)
    """
    dataset_cls = {
        "MNIST": MNIST,
        "FMNIST": FashionMNIST,
        "CIFAR10": CIFAR10,
        #"TinyImageNet": TinyImageNet,
        "CICADA": CICADA,
    }.get(base_name)

    if dataset_cls is None:
        raise ValueError(f"Unknown dataset name: {base_name}")
    
    dataset_kwargs = {
        "root": root,
        "download": True,
        "transform": CicadaTransform() if base_name == "CICADA" else transforms.ToTensor(),
    }

    train_ds = dataset_cls(**dataset_kwargs, train=True)
    val_ds = dataset_cls(**dataset_kwargs, train=False)

    return train_ds, val_ds


def get_exposure_datasets(image_shape: tuple, n_train: int, n_val: int, exposure_method="noise", teacher=None):
    """
    Get the exposure datasets for the specified exposure method.
    :param image_shape: Shape of the input data
    :param n_train: Number of training samples
    :param n_val: Number of validation samples
    :param exposure_method: Method to generate exposure data (e.g., 'noise')
    :param teacher: Only used for langevin sampling
    :return: Exposure datasets
    """
    ds_kwargs = {
        # "transform": transforms.ToTensor(),
    }
    if exposure_method == "noise":
        train_ds = NoiseDataset(image_size=image_shape, size=n_train, **ds_kwargs)
        val_ds = NoiseDataset(image_size=image_shape, size=n_val, **ds_kwargs)
    elif exposure_method == "one":
        train_ds = OneDataset(image_size=image_shape, size=n_train, **ds_kwargs)
        val_ds = OneDataset(image_size=image_shape, size=n_val, **ds_kwargs)
    else:
        raise ValueError(f"Unknown exposure method: {exposure_method}")
    return train_ds, val_ds


class InlierSampler(torch.utils.data.Sampler):
    """
    Samples only from inlier indices
    Assumes label 0/1 for in-/outliers
    """
    def __init__(self, inlier_indices, shuffle=False, max_n=None):
        self.inlier_indices = inlier_indices
        
        # Limit the number of samples if max_n is specified
        if max_n is not None and max_n < len(self.inlier_indices):
            self.inlier_indices = self.inlier_indices[:max_n]
            
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            indices = self.inlier_indices.copy()
            np.random.shuffle(indices)
            return iter(indices)
        return iter(self.inlier_indices)
        
    def __len__(self):
        return len(self.inlier_indices)


def get_loaders(
    ds_name: str,
    hold_out_classes: Union[int, set, list],
    batch_size: int = 1024,
    n_max: int = None,
    root: str = './data',
    shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get the train and test loaders for the specified dataset and holdout classes.
    :param ds_name: Name of the dataset (e.g., 'MNIST', 'FMNIST', 'CIFAR10', 'TinyImageNet')
    :param hold_out_classes: Classes to be treated as outliers (single or multiple ints)
    :param batch_size: Batch size for the DataLoader
    :param n_max: Maximum number of inlier samples to use
    :return: Train and val DataLoader objects
    """
    train_ds, test_ds = get_base_datasets(ds_name, root=root)

    hold_out_set = {hold_out_classes} if isinstance(hold_out_classes, int) else set(hold_out_classes)

    if ds_name == "CICADA":
        # all_labels = np.unique(train_ds.targets.numpy())
        all_labels = np.unique(train_ds.targets)
        hold_out_set = set(all_labels) - {0}
        print(f"Overriding hold-out set for CICADA to {hold_out_set}. (Training on ZB)")

    train_ds.target_transform = lambda x: (x in hold_out_set)
    test_ds.target_transform = lambda x: (x in hold_out_set)

    # train_ds is size limited by sampler, test_ds directly here
    if n_max is not None:
        test_ds = Subset(test_ds, range(n_max))
    
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }

    inlier_indices = get_inlier_inidices(train_ds.targets, hold_out_set)
    sampler = InlierSampler(inlier_indices, shuffle=shuffle, max_n=n_max)

    train_loader = DataLoader(train_ds, **loader_kwargs, sampler=sampler)
    test_loader = DataLoader(test_ds, **loader_kwargs, shuffle=shuffle)
    
    return train_loader, test_loader


def get_target_loaders(
    ds_name: str,
    hold_out_classes: Union[int, set, list],
    teacher: torch.nn.Module,
    distillation_encoder: callable = lambda x: torch.log1p(x),
    exposure_method: str = "noise",
    exposure_fraction: float = 0.1,
    batch_size: int = 1024,
    n_max: int = None,
    root: str = './data',
    shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get the train and test loaders for the specified dataset and holdout classes.
    :param ds_name: Name of the dataset (e.g., 'MNIST', 'FMNIST', 'CIFAR10', 'TinyImageNet')
    :param hold_out_classes: Classes to be treated as outliers (single or multiple ints)
    :param teacher: The teacher model that generates the soft targets (and potentially the exposure data)
    :param distillation_encoder: Mapping from teacher's reconstruction loss to soft targets
    :param exposure_method: Method to generate exposure data (e.g., 'noise', 'outlier', 'langevin')
    :param exposure_fraction: Fraction of exposure data to add to the dataset (proportionally same for train and val)
    :param batch_size: Batch size for the DataLoader
    :param n_max: Maximum number of inlier samples to use
    :param root: Root directory for the dataset
    :param shuffle: Whether to shuffle the data
    :return: Train and val DataLoader objects
    """
    hold_out_set = {hold_out_classes} if isinstance(hold_out_classes, int) else set(hold_out_classes)

    def target_transform(x):
        with torch.no_grad():
            reco_error = teacher.predict(x)
            soft_targets = distillation_encoder(reco_error)
        return soft_targets

    base_train_ds, base_val_ds = get_base_datasets(ds_name, root=root)

    train_inlier_indices = get_inlier_inidices(base_train_ds.targets, hold_out_set)
    n_train, n_val = len(train_inlier_indices), len(base_val_ds)

    exposure_train_ds, exposure_val_ds = get_exposure_datasets(
        image_shape=base_train_ds[0][0].shape,
        n_train=n_train,
        n_val=n_val,
        exposure_method=exposure_method,
        teacher=teacher,
    )

    train_ds = torch.utils.data.ConcatDataset([base_train_ds, exposure_train_ds])
    val_ds = torch.utils.data.ConcatDataset([base_val_ds, exposure_val_ds])

    train_ds = TargetDataset(train_ds, teacher_mapping=teacher.predict, distillation_encoder=distillation_encoder)
    val_ds = TargetDataset(val_ds, teacher_mapping=teacher.predict, distillation_encoder=distillation_encoder)
    
    # Consider all exposure data, but only inliers from the base dataset for training
    train_inlier_indices = np.concatenate([train_inlier_indices, np.arange(len(exposure_train_ds))+len(base_train_ds)])
    train_sampler = InlierSampler(train_inlier_indices, shuffle=True, max_n=n_max)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": True,
    }

    train_loader = DataLoader(train_ds, **loader_kwargs, sampler=train_sampler)
    val_loader = DataLoader(val_ds, **loader_kwargs, shuffle=shuffle)

    return train_loader, val_loader
