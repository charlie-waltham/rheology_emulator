import pickle
import torch
import logging
from pathlib import Path

import xarray as xr
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split, Subset

from ..scaling import LogFLTransformer


class TorchDataManager:
    def __init__(self, arguments: dict):
        # Initialize the data manager with the file path and arguments
        self.pairs_path = arguments.get("pairs_path")
        self.batch_size = arguments.get("batch_size")
        self.val_fraction = arguments.get("val_fraction")
        self.test_fraction = arguments.get("test_fraction")
        self.scale = arguments.get("scale_features")
        self.scale_factor_features = 20
        self.scale_factor_labels = 100
        self.train_features = arguments.get("train_features")
        self.train_labels = arguments.get("train_labels")
        self.difference_labels = arguments.get("difference_labels")
        self.shorten_dataset = arguments.get("shorten_dataset")
        self.sequential = arguments.get("sequential")

        # load the raw data
        self.raw_data = self._load_zarr()

        # if difference labels is true, difference the labels
        if self.difference_labels:
            differenced = self.raw_data.get("d_label")
            if differenced is not None:
                logging.info("Using pre-differenced labels from d_labels")
                self.raw_data["labels"].values = self.raw_data["d_labels"].values
            else:
                self._difference_labels()

        # extract numerical data for features and labels, subsetting according to train_features and train_labels
        self.features, self.labels = self._extract_features_labels()

        if arguments.get("test"):
            self.eval_path = Path(arguments["eval_path"])
            # Get scaler
            scaler_path = self.eval_path / "data_splits/scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as scaler_file:
                    self.scaler = pickle.load(scaler_file)

                # Scale by precalculated scaler
                self.features = self.scaler.feature_scaler.transform(self.features)
                self.labels = self.scaler.label_scaler.transform(self.labels)

            indices_path = self.eval_path / "data_splits/test_dataset.pt"
            if not indices_path.exists():
                logging.error(f"Test dataset indices not found at {indices_path}")
                return

            self.dataset = FeatureLabelDataset(self.features, self.labels)
            test_indices = torch.load(indices_path, weights_only=False)
            test_dataset = Subset(self.dataset, test_indices)

            self.test_loader = self._standard_dataloader(test_dataset, shuffle=False)

            self.n_test = len(self.test_loader.dataset)
            self.n_batches_test = len(self.test_loader)
            self.n_features = len(self.train_features)
            self.n_labels = len(self.train_labels)

        else:
            if self.scale:
                self._scale()

            # make a torch style dataset
            self.dataset = FeatureLabelDataset(self.features, self.labels)

            # if we need a shorter training set, for debugging etc
            if self.shorten_dataset is not None:
                if not self.sequential:
                    indices = np.random.choice(
                        len(self.dataset), size=self.shorten_dataset, replace=False
                    )
                else:
                    indices = np.arange(self.shorten_dataset)
                self.dataset = torch.utils.data.Subset(self.dataset, indices)

            # make data loaders for training, validation, and testing
            self.train_loader, self.val_loader, self.test_loader = self._make_loaders()

            # assign dimensions of loaders
            self._get_loader_sizes()

        self.raw_data.close()

    def _load_zarr(self):
        return xr.open_zarr(self.pairs_path)

    def _difference_labels(self):
        logging.info("Differencing labels...")
        self.raw_data["labels"].loc[dict(label="sivelv")] = self.raw_data["labels"].sel(
            label="sivelv"
        ) - self.raw_data["features"].sel(feature="sivelv")
        self.raw_data["labels"].loc[dict(label="sivelu")] = self.raw_data["labels"].sel(
            label="sivelu"
        ) - self.raw_data["features"].sel(feature="sivelu")
        logging.info("Labels differenced.")

    def _extract_features_labels(self):
        features = self.raw_data["features"].sel(feature=self.train_features).values.T
        labels = self.raw_data["labels"].sel(label=self.train_labels).values.T
        return features, labels

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
        self.scaler = LogFLTransformer(
            self.features,
            self.labels,
            self.train_features,
            self.scale_factor_features,
            self.scale_factor_labels,
        )

        self.features = self.scaler.feature_scaler.transform(self.features)
        self.labels = self.scaler.label_scaler.transform(self.labels)

        feature_vel_scale = (
            self.scaler.feature_scaler.named_transformers_["velocity"]
            .named_steps["scaler"]
            .scale_
        )
        label_vel_scale = self.scaler.label_scaler.named_steps["scaler"].scale_
        logging.info(
            f"Feature vel scale: {feature_vel_scale}\nLabel vel scale: {label_vel_scale}"
        )

    def save_datasets(self, save_path):
        # Save dataset indices
        torch.save(self.train_loader.dataset.indices, save_path + "train_dataset.pt")
        torch.save(self.val_loader.dataset.indices, save_path + "val_dataset.pt")
        torch.save(self.test_loader.dataset.indices, save_path + "test_dataset.pt")
        if self.scale:
            with open(save_path + "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
        print(f"Datasets saved to {save_path}")


# Define PyTorch Dataset
class FeatureLabelDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
