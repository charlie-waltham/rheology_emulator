import pickle
import math
import torch

import xarray as xr
import pandas as pd

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split

class TorchDataManager:
    def __init__(self, file_path, arguments=None, zarr_fmt='fmt1', difference_labels=False):
        # Initialize the data manager with the file path and arguments
        self.file_path = file_path
        self.batch_size = arguments['batch_size']
        self.val_fraction = arguments['val_fraction']
        self.test_fraction = arguments['test_fraction']
        self.scale = arguments['scale_features']
        self.train_features = arguments['train_features']
        self.train_labels = arguments['train_labels']
        self.zarr_fmt = zarr_fmt
        self.difference_labels = difference_labels

        # Load the data from the specified file path
        if self.zarr_fmt == 'fmt1':
            self.raw_data = self._load_zarr_1()
            self.pairs = FeatureLabelDataset(self.raw_data)
        elif self.zarr_fmt == 'fmt2':
            self.raw_data = self._load_zarr_2()
            self.pairs = FeatureLabelDataset_2(self.raw_data)
        # else:
        #     self.raw_data = self._load()

        # if self.difference_labels:
        #     self._make_labels_differenced()


        # Get loaders for training, validation, and testing
        self.train, self.val, self.test = self._get_data_loaders()

        # Print short summary
        self._print_summary()
        
    def _print_summary(self):
        print(f"Data loaded from {self.file_path}")
        print(f"Data from zarr {self.zarr_fmt} format")
        print(f"Batch size: {self.batch_size}")
        print(f"Validation fraction: {self.val_fraction}")
        print(f"Test fraction: {self.test_fraction}")
        print(f"Dataset sizes: Train={len(self.train.dataset)}, Val={len(self.val.dataset)}, Test={len(self.test.dataset)}")
    

    # def _load(self):
    #     # with open(self.file_path, 'rb') as f:
    #     #     return pickle.load(f)
        
    #     with open(self.file_path, 'rb') as f:
    #         pairs = pickle.load(f)

    #     # Get the required features and labels, and convert to a numpy array
    #     pairs = self._extract_features_labels(pairs)

    #     return pairs
    
    def _load_zarr_1(self):

        pairs = xr.open_zarr(self.file_path)

        # Process zarr
        # TODO: Does this belong here, or in data_manager, or in a new function below?
        # TODO: add specifiable region masking
        arctic_mask = (pairs.lat > 60).compute()
        pairs = pairs.where(arctic_mask, drop=True).compute()
        # nan siconc=0 data
        siconc_mask = (pairs['features'].sel(feature='siconc') > 0).compute()
        # broadcast to features and labels
        mask_features = siconc_mask.expand_dims({'feature': pairs['features'].feature}, axis=1)
        mask_labels = siconc_mask.expand_dims({'label': pairs['labels'].label}, axis=1)
        pairs['features'] = pairs['features'].where(mask_features)
        pairs['labels'] = pairs['labels'].where(mask_labels)
        # stack spatial
        pairs = pairs.stack(xy=('x','y'))
        # drop nan
        # TODO: check the behaviour of this line. is it dropping any real data?
        pairs = pairs.dropna(dim='xy')

        # TODO: add specifiable features and labels selection
        features = pairs['features'].sel(feature=self.train_features)
        labels = pairs['labels'].sel(label=self.train_labels)

        if self.difference_labels:
            for ilabel, label in enumerate(self.train_labels):
                if label in self.train_features:
                    new_label_values = labels.sel(label=label).values - features.sel(feature=label).values
                    labels.loc[dict(label=label)] = xr.DataArray(
                            new_label_values,
                            dims=('pair', 'xy'),
                            coords={'pair': labels.pair, 'xy': labels.xy}
                        )

        pd_pairs = []
        for pair in range(features.shape[0]):
            fl = (
                features[pair].values.T,
                labels[pair].values.T
            )
            pd_pairs.append(fl)

        return pd_pairs
    
    def _load_zarr_2(self):
        
        pairs = xr.open_zarr(self.file_path)

        # Process zarr
        features = pairs.features.sel(feature=self.train_features)
        labels = pairs.labels.sel(label=self.train_labels)

        # TODO: need to implement differencing here
        if self.difference_labels:
            pass

        pd_features = pd.DataFrame(features.values.T, columns=self.train_features)
        pd_labels = pd.DataFrame(labels.values.T, columns=self.train_labels)

        return (pd_features.values, pd_labels.values)

    def _make_labels_differenced(self):
        # Subtract features from labels for all pairs
        pass

    def _extract_features_labels(self, pairs):
        # Subset to the desired features and labels
        for ipair, pair in enumerate(pairs):
            features = pair[0][self.train_features]
            labels = pair[1][self.train_labels]
            pairs[ipair] = [features.values, labels.values]

        return pairs

    
    def _scale_features(self, pairs):
        scaler = StandardScaler()
        # Fit the scaler on the features
        # TODO: this will NOT work with fmt2, need to think about how to do this properly
        scaler.fit(pairs[0][0])  # TODO: This assumes first feature is representative, so need to think about this
        # Transform the features
        scaled_features = [scaler.transform(features) for features, _ in pairs]
        # Return pairs with scaled features without converting to torch tensor
        self.scaler = scaler
        pairs = [(scaled_features[i], label) for i, (_, label) in enumerate(pairs)]

    def _make_torch_tensors(self, pairs):
        # Convert data to PyTorch tensors
        tensor_data = [(torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)) for features, label in pairs]
        return tensor_data

    def _split_dataset(self):
        total = len(self.dataset)
        test_len = math.floor(total * self.test_fraction)
        val_len = math.floor(total * self.val_fraction)
        train_len = total - val_len - test_len
        return random_split(self.dataset, [train_len, val_len, test_len])

    def _get_data_loaders(self):

        # Scale data, or not
        if self.scale:
            self._scale_features(self.pairs)

        # Convert pairs to PyTorch tensors
        self.dataset = self._make_torch_tensors(self.pairs)

        # Split the dataset into training, validation, and test sets
        train_set, val_set, test_set = self._split_dataset()

        # Create DataLoader objects for each set
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

class FeatureLabelDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        return features, label
    
class FeatureLabelDataset_2(Dataset):
    def __init__(self, data):
        self.features, self.labels = data
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def make_training_data_from_pairs(pairs, arguments):
    pass

def make_data_loaders(dataset, batch_size, val_fraction, test_fraction):
    pass
