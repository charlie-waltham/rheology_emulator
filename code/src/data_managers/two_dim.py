import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from ..scaling import InvertableColumnTransformer


def _log_scaled(x, scale_factor):
    return np.sign(x) * np.log1p(np.abs(x) * scale_factor)


def _exp_scaled(x, scale_factor):
    return np.sign(x) * (np.expm1(np.abs(x)) / scale_factor)


class TorchDataManager:
    def __init__(self, arguments: dict):
        self.pairs_path = arguments.get("pairs_path")
        self.cache_path = arguments.get("cache_path")
        self.batch_size = arguments.get("batch_size")
        self.val_fraction = arguments.get("val_fraction")
        self.test_fraction = arguments.get("test_fraction")
        self.scaling = arguments.get("scaling")
        self.scale_factor = arguments.get("scale_factor")
        self.train_features = arguments.get("train_features")
        self.train_labels = arguments.get("train_labels")
        self.vectors = arguments.get("vectors")
        self.difference_labels = arguments.get("difference_labels")
        self.sequential = arguments.get("sequential")
        self.patch_size = arguments.get("patch_size")
        self.patches_per_image = arguments.get("patches_per_image")

        self.mask = xr.open_dataset("../data/weddell_mask.nc")["tmask"].values
        self.raw_data = self._load()
        self.train_features = list(filter(lambda x: not x.startswith("label"), self.raw_data.data_vars.keys()))

        self.features: np.ndarray = self.raw_data[self.train_features].to_dataarray().values.T
        self.labels: np.ndarray = self.raw_data[["label_" + var for var in self.train_labels]].to_dataarray().values.T
        self.raw_data.close()

        if arguments.get("test"):
            self.eval_path = Path(arguments["eval_path"])
            if self.scaling is not None:
                self._scale_test()

            indices_path = self.eval_path / "data_splits/test_dataset.pt"
            if not indices_path.exists():
                logging.error(f"Test dataset indices not found at {indices_path}")
                return

            self.dataset = FeatureLabelDataset(self.features, self.labels, self.mask)
            test_indices = torch.load(indices_path, weights_only=False)
            test_dataset = Subset(self.dataset, test_indices)

            self.test_loader = self._standard_dataloader(test_dataset, shuffle=False)

            self.n_test = len(self.test_loader.dataset)
            self.n_batches_test = len(self.test_loader)
            self.n_features = len(self.train_features)
            self.n_labels = len(self.train_labels)

        else:
            if self.scaling is not None:
                self._scale()

            # make a torch style dataset
            self.dataset = FeatureLabelDataset(self.features, self.labels, self.mask)

            # make data loaders for training, validation, and testing
            self.train_loader, self.val_loader, self.test_loader = self._make_loaders()

            # assign dimensions of loaders
            self._get_loader_sizes()

    def _load(self) -> xr.Dataset:
        if self.cache_path is not None:
            if os.path.exists(self.cache_path):
                logging.info("Cached file exists. Loading")
                return xr.open_dataset(self.cache_path)
            logging.info("No cached file found. Generating")

        ds = xr.open_mfdataset(self.pairs_path, data_vars="all")
        ds = ds[self.train_features]

        if self.vectors is not None:
            for vector, components in self.vectors.items():
                values = ds[components].to_dataarray()
                magnitude = xr.ufuncs.hypot(values[0], values[1])
                direction = values / (magnitude + 1e-8)

                ds[vector + "_mag"] = magnitude
                ds[vector + "_u"] = direction[0]
                ds[vector + "_v"] = direction[1]
                ds = ds.drop_vars(components)

        if self.difference_labels:
            # label = var_{t+1} - var_t
            ds: xr.Dataset = ds.assign({
                "label_" + var: ds[var].shift(time_counter=-1) - ds[var] for var in self.train_labels
            })
        else:
            # label = var_{t+1}
            ds: xr.Dataset = ds.assign({
                "label_" + var: ds[var].shift(time_counter=-1) for var in self.train_labels
            })

        # Remove last timestep as no label
        ds = ds.fillna(0).isel(time_counter=slice(-1))

        if self.cache_path is not None:
            ds.to_netcdf(self.cache_path)

        return ds

    def _make_loaders(self):
        if not self.sequential:
            total_size = len(self.dataset)
            test_size = int(total_size * self.test_fraction)
            val_size = int(total_size * self.val_fraction)
            train_size = total_size - test_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset, [train_size, val_size, test_size]
            )
        else:
            # Need to produce datasets manually to preserve order
            values = np.random.rand(len(self.dataset))
            test_mask = values < self.test_fraction
            val_mask = (
                values
                < self.test_fraction + self.val_fraction & values
                >= self.test_fraction
            )
            train_mask = values >= self.test_fraction + self.val_fraction

            train_dataset = self.dataset[train_mask]
            val_dataset = self.dataset[val_mask]
            test_dataset = self.dataset[test_mask]

        if self.patch_size is not None:
            train_dataset = RandomPatchDataset(train_dataset, self.patch_size, self.patches_per_image)

        train_loader = self._standard_dataloader(
            train_dataset, shuffle=not self.sequential
        )
        val_loader = self._standard_dataloader(val_dataset, shuffle=False)
        test_loader = self._standard_dataloader(test_dataset, shuffle=False)

        return train_loader, val_loader, test_loader

    def _standard_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def _get_loader_sizes(self):
        self.n_train = len(self.train_loader.dataset)
        self.n_val = len(self.val_loader.dataset)
        self.n_test = len(self.test_loader.dataset)
        self.n_batches_train = len(self.train_loader)
        self.n_batches_val = len(self.val_loader)
        self.n_batches_test = len(self.test_loader)
        self.n_features = len(self.train_loader.dataset[0][0])
        self.n_labels = len(self.train_loader.dataset[0][1])

    def _scale(self):
        flat_size = np.multiply.reduce(self.features.shape[:3])
        features_shape = self.features.shape
        labels_shape = self.labels.shape

        # Need to temporarily flatten non-feature dimensions for sklearn transformers
        # From (X, Y, time, features) to (samples, features)
        flattened_features = self.features.reshape((flat_size, features_shape[-1]))
        flattened_labels = self.labels.reshape((flat_size, labels_shape[-1]))

        feature_log_indices = self._indices_from_names(self.train_features, self.scaling.get("log"))
        feature_std_indices = self._indices_from_names(self.train_features, self.scaling.get("std"))
        label_log_indices = self._indices_from_names(self.train_labels, self.scaling.get("log"))
        label_std_indices = self._indices_from_names(self.train_labels, self.scaling.get("std"))

        self.feature_scaler = InvertableColumnTransformer(
            transformers=self._build_transformers(feature_log_indices, feature_std_indices),
            remainder="passthrough",
        )
        self.label_scaler = InvertableColumnTransformer(
            transformers=self._build_transformers(label_log_indices, label_std_indices),
            remainder="passthrough",
        )

        self.features = self.feature_scaler.fit_transform(flattened_features).reshape(features_shape)
        self.labels = self.label_scaler.fit_transform(flattened_labels).reshape(labels_shape)

    def _make_log_scaler(self):
        return Pipeline([
            ("log", FunctionTransformer(
                    func=_log_scaled,
                    inverse_func=_exp_scaled,
                    validate=True,
                    check_inverse=False,
                    kw_args={"scale_factor": self.scale_factor},
                    inv_kw_args={"scale_factor": self.scale_factor},
                ),
            ),
            ("scaler", RobustScaler()),
        ])

    def _build_transformers(self, log_indices, std_indices):
        transformers = []
        if log_indices:
            transformers.append(("log", self._make_log_scaler(), log_indices))
        if std_indices:
            transformers.append(("std", RobustScaler(), std_indices))
        return transformers

    def _indices_from_names(self, index, names):
        if names is None:
            return None
        return [index.index(n) for n in names if n in index]

    def _scale_test(self):
        feature_scaler_path = self.eval_path / "data_splits/feature_scaler.pkl"
        label_scaler_path = self.eval_path / "data_splits/label_scaler.pkl"

        if feature_scaler_path.exists() and label_scaler_path.exists():
            flat_size = np.multiply.reduce(self.features.shape[:3])
            features_shape = self.features.shape
            labels_shape = self.labels.shape
            flattened_features = self.features.reshape((flat_size, features_shape[-1]))
            flattened_labels = self.labels.reshape((flat_size, labels_shape[-1]))

            with open(feature_scaler_path, "rb") as f:
                self.feature_scaler = pickle.load(f)
            with open(label_scaler_path, "rb") as f:
                self.label_scaler = pickle.load(f)

            self.features = self.feature_scaler.transform(flattened_features).reshape(features_shape)
            self.labels = self.label_scaler.transform(flattened_labels).reshape(labels_shape)

    def save_datasets(self, save_path):
        # Save dataset indices
        train_indices = self.train_loader.dataset.indices if self.patch_size is None else self.train_loader.dataset.dataset.indices
        torch.save(train_indices, save_path + "train_dataset.pt")
        torch.save(self.val_loader.dataset.indices, save_path + "val_dataset.pt")
        torch.save(self.test_loader.dataset.indices, save_path + "test_dataset.pt")
        if self.scaling is not None:
            with open(save_path + "feature_scaler.pkl", "wb") as f:
                pickle.dump(self.feature_scaler, f)
            with open(save_path + "label_scaler.pkl", "wb") as f:
                pickle.dump(self.label_scaler, f)
        print(f"Datasets saved to {save_path}")


class FeatureLabelDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, mask: np.ndarray = None):
        self.features = torch.tensor(features.transpose((2, 3, 0, 1)), dtype=torch.float32)
        self.labels = torch.tensor(labels.transpose((2, 3, 0, 1)), dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.mask

class RandomPatchDataset(Dataset):
    """
    Wraps a PyTorch Dataset to extract random spatial patches.
    Artificially expands the virtual size of the dataset to ensure adequate 
    spatial coverage during an epoch.
    """
    def __init__(self, subset_dataset, patch_size: int, patches_per_image: int = 10):
        self.dataset = subset_dataset
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

    def __len__(self):
        # Inflate the length so the DataLoader iterates longer per epoch
        return len(self.dataset) * self.patches_per_image

    def __getitem__(self, idx):
        # Map the inflated index back to the actual image index
        real_idx = idx // self.patches_per_image
        feat, lab, msk = self.dataset[real_idx]

        # Get spatial dimensions (assuming shape is C, H, W)
        _, h, w = feat.shape
        
        # Ensure patch size isn't somehow larger than the image
        p_h = min(self.patch_size, h)
        p_w = min(self.patch_size, w)

        # Randomly select the top-left corner of the patch
        top = torch.randint(0, h - p_h + 1, (1,)).item()
        left = torch.randint(0, w - p_w + 1, (1,)).item()

        # Slice the patch across spatial dimensions
        feat_patch = feat[:, top:top+p_h, left:left+p_w]
        lab_patch = lab[:, top:top+p_h, left:left+p_w]
        msk_patch = msk[top:top+p_h, left:left+p_w]

        return feat_patch, lab_patch, msk_patch
