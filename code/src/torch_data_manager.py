import pickle
import torch
import logging

import xarray as xr
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, FunctionTransformer
from torch.utils.data import Dataset, DataLoader, random_split

from .invertable_column_transformer import InvertableColumnTransformer

class TorchDataManager:
    def __init__(self, file_path, arguments, difference_labels=False):
        # Initialize the data manager with the file path and arguments
        self.file_path = file_path
        self.batch_size = arguments['batch_size']
        self.val_fraction = arguments['val_fraction']
        self.test_fraction = arguments['test_fraction']
        self.scale = arguments['scale_features']
        self.scale_factor_features = 10
        self.scale_factor_labels = 100
        self.train_features = arguments['train_features']
        self.train_labels = arguments['train_labels']
        self.difference_labels = difference_labels
        self.shorten_dataset = arguments['shorten_dataset']
        self.sequential = arguments['sequential']

        # load the raw data
        self.raw_data = self._load_zarr()

        # if difference labels is true, difference the labels
        if self.difference_labels:
            self._difference_labels()

        # extract numerical data for features and labels, subsetting according to train_features and train_labels
        self.features, self.labels = self._extract_features_labels()

        # scale velocity by an arcsinh scaler, rest by minmax scaler
        if self.scale:
            self._scale()

        # make a torch style dataset
        self.dataset = FeatureLabelDataset(self.features, self.labels)

        # if we need a shorter training set, for debugging etc
        if self.shorten_dataset is not None:
            if not self.sequential:
                indices = np.random.choice(len(self.dataset), size=self.shorten_dataset, replace=False)
            else:
                indices = np.arange(self.shorten_dataset)
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

        # make data loaders for training, validation, and testing
        self.train_loader, self.val_loader, self.test_loader = self._make_loaders()

        # assign dimensions of loaders
        self._get_loader_sizes()

        # Print short summary
        self._print_summary()
        
    def _print_summary(self):
        print(f"Data loaded from {self.file_path}")
        print(f"Data from zarr fmt2 format (long list)")
        print(f"Batch size: {self.batch_size}")
        print(f"Validation fraction: {self.val_fraction}")
        print(f"Test fraction: {self.test_fraction}")
        print(f"Dataset sizes: Train={self.n_train}, Val={self.n_val}, Test={self.n_test}")
        print(f"Number of batches: Train={self.n_batches_train}, Val={self.n_batches_val}, Test={self.n_batches_test}")
        print(f"Number of features: {self.n_features}")
        print(f"Number of labels: {self.n_labels}")

    def _load_zarr(self):
        return xr.open_zarr(self.file_path)
    
    def _difference_labels(self):
        logging.info('Differencing labels...')
        self.raw_data['labels'].loc[dict(label='sivelv')] = self.raw_data['labels'].sel(label='sivelv') - self.raw_data['features'].sel(feature='sivelv')
        self.raw_data['labels'].loc[dict(label='sivelu')] = self.raw_data['labels'].sel(label='sivelu') - self.raw_data['features'].sel(feature='sivelu')
        logging.info('Labels differenced.')

    def _extract_features_labels(self):
        features = self.raw_data['features'].sel(feature=self.train_features).values.T
        labels = self.raw_data['labels'].sel(label=self.train_labels).values.T
        return features, labels
    
    def _make_loaders(self):
        total_size = len(self.dataset)
        test_size = int(total_size * self.test_fraction)
        val_size = int(total_size * self.val_fraction)
        train_size = total_size - test_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

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
        # Define a custom scaler for velocity values
        self.feature_vel_scaler = Pipeline([
            ('log', FunctionTransformer(func=self._log_scaled,
                                            inverse_func=self._exp_scaled,
                                            validate=True,
                                            kw_args={'scale_factor': self.scale_factor_features},
                                            inv_kw_args={'scale_factor': self.scale_factor_features})),
            ('scaler', MaxAbsScaler())
        ])
        self.label_vel_scaler = Pipeline([
            ('log', FunctionTransformer(func=self._log_scaled,
                                            inverse_func=self._exp_scaled,
                                            validate=True,
                                            kw_args={'scale_factor': self.scale_factor_labels},
                                            inv_kw_args={'scale_factor': self.scale_factor_labels})),
            ('scaler', MaxAbsScaler())
        ])

        # Find which velocity values are present
        vel_indices = [self.train_features.index(n) for n in ['sivelv', 'sivelu'] if n in self.train_features]

        # Define a scaler for all features
        self.scaler = InvertableColumnTransformer(
            transformers=[
                ('velocity', self.feature_vel_scaler, vel_indices)
            ],
            remainder=MinMaxScaler()
        )

        self.features = self.scaler.fit_transform(self.features)
        self.labels = self.label_vel_scaler.fit_transform(self.labels)

        feature_vel_scale = self.scaler.named_transformers_['velocity'].named_steps['scaler'].scale_
        label_vel_scale = self.label_vel_scaler.named_steps['scaler'].scale_
        logging.info(f'Feature vel scale: {feature_vel_scale}\nLabel vel scale: {label_vel_scale}')

    def _arcsinh_scaled(self, x, scale_factor=2):
        return np.arcsinh(x * scale_factor)
    def _sinh_scaled(self, x, scale_factor=2):
        return np.sinh(x) / scale_factor
    
    def _log_scaled(self, x, scale_factor=1):
        return np.sign(x) * np.log1p(np.abs(x) * scale_factor)
    def _exp_scaled(self, x, scale_factor=1):
        return np.sign(x) * (np.expm1(np.abs(x)) / scale_factor)

    def save_datasets(self, save_path):
        # Save the datasets to the specified path
        torch.save(self.train_loader.dataset, save_path + 'train_dataset.pt')
        torch.save(self.val_loader.dataset, save_path + 'val_dataset.pt')
        torch.save(self.test_loader.dataset, save_path + 'test_dataset.pt')
        if self.scale:
            with open(save_path + 'scaler.pkl', 'wb') as f:
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