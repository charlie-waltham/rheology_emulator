import torch
import yaml
import logging
import pickle

from . import utils

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# import ordinary least squares regression
from sklearn.linear_model import LinearRegression

def setup_logging(log_file='train_nn.log'):
    """
    Set up logging to a file.
    """
    import logging
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging setup complete.")

class NNCapsule:
    def __init__(self, arguments):
        self.arguments = arguments

        # Load data
        self.zarr_fmt = arguments['zarr_fmt']
        if self.zarr_fmt == 'fmt1':
            from .dataset_utils import TorchDataManager
            self.data_manager = TorchDataManager(arguments['pairs_path'], arguments, zarr_fmt=self.zarr_fmt, difference_labels=arguments['difference_labels'])
            self.train_loader = self.data_manager.train.dataset
            self.val_loader = self.data_manager.val.dataset
            self.n_features = self.train_loader[0][0].shape[1]
            self.n_labels = self.train_loader[0][1].shape[1]
            self.n_batches = len(self.train_loader)
            self.n_samples = len(self.train_loader.dataset)
            self.n_observations = self.train_loader[10][0].shape[0]
        elif self.zarr_fmt == 'fmt2':
            from .torch_data_manager import TorchDataManager
            self.data_manager = TorchDataManager(arguments['pairs_path'], arguments, difference_labels=arguments['difference_labels'])
            self.n_features = self.data_manager.n_features
            self.n_labels = self.data_manager.n_labels
            self.n_samples = self.data_manager.n_train
            self.n_batches = self.data_manager.n_batches_train
            self.train_loader = self.data_manager.train_loader
            self.val_loader = self.data_manager.val_loader
            self.test_loader = self.data_manager.test_loader

        self.scaler = self.data_manager.scaler
        self.label_scaler = self.data_manager.label_vel_scaler

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")  # Force CPU for debugging

        # Define model
        self.architecture = arguments['architecture']
        #self.parameters = arguments['parameters']
        # TODO: this is clunky and params should be wrapped up into arguments
        self.parameters = '../configs/training/' + arguments['training_cfg'] + '.yaml'
        self.model = utils.define_nn(self.architecture, self.n_features, self.n_labels, self.device)
        # TODO: split the below up so that they're called separately, or do some order agnostic unpacking of all the parameters
        self.criterion, self.optimizer, self.n_epochs = utils.nn_options(self.model, self.parameters)
        self.train_losses = []
        self.val_losses = []

        # Set up logging
        setup_logging(log_file='train_nn.log')
        self._print_summary()

    def _print_summary(self):
        logging.info("Model Summary:")
        logging.info(f"Architecture: {self.architecture}")
        logging.info(f"Parameters: {self.parameters}")
        logging.info(f"Number of training samples: {self.n_samples}")
        #logging.info(f"Number of data points in sample: {self.n_observations}")
        logging.info(f"Number of batches: {self.n_batches}")
        logging.info(f"Number of features: {self.n_features}")
        logging.info(f"Number of labels: {self.n_labels}")

    def train(self):
        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.train_losses.append(running_loss / len(self.train_loader))

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            self.val_losses.append(val_loss / len(self.val_loader))

            logging.info(f"Epoch {epoch+1}, Train Loss: {self.train_losses[-1]:.2e}, Val Loss: {self.val_losses[-1]:.2e}")

        logging.info("Training complete.")
        logging.info(f"Final Train Loss: {self.train_losses[-1]:.2e}, Final Val Loss: {self.val_losses[-1]:.2e}")

    def plot_train_losses(self, train_losses, val_losses):

        #ylims = [0, np.median(train_losses)*4]

        fig = plt.figure(figsize=(5, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        #plt.ylim(ylims)
        plt.legend()
        # TODO: replace show with some saving option
        plt.savefig(self.arguments['results_path'] + 'train_losses.png')
        logging.info("Training losses plotted and saved.")
        return fig
    
    def ytrue_ypred(self, loader):
        predictions = []
        true_values = []
        with torch.no_grad():  # Disable gradient tracking
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs)
                true_values.append(targets)

        # Concatenate all batches into single tensors
        predictions = torch.cat(predictions, dim=0).to("cpu")
        true_values = torch.cat(true_values, dim=0).to("cpu")

        # Unscale the true values and predictions
        if (self.data_manager.scale):
            predictions = self.label_scaler.inverse_transform(predictions)
            true_values = self.label_scaler.inverse_transform(true_values)

            predictions = torch.tensor(predictions)
            true_values = torch.tensor(true_values)

        return true_values, predictions
    
    def save_ytrue_ypred_inputs(self, loader, path):
        predictions = []
        true_values = []
        inputs_list = []
        with torch.no_grad():  # Disable gradient tracking
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs)
                true_values.append(targets)
                inputs_list.append(inputs)

        # Concatenate all batches into single tensors
        predictions = torch.cat(predictions, dim=0).to("cpu")
        true_values = torch.cat(true_values, dim=0).to("cpu")
        inputs_all = torch.cat(inputs_list, dim=0).to("cpu")

        # Unscale the true values, predictions and inputs
        if (self.data_manager.scale):
            predictions = torch.tensor(self.label_scaler.inverse_transform(predictions))
            true_values = torch.tensor(self.label_scaler.inverse_transform(true_values))
            inputs_all = self.scaler.inverse_transform(inputs_all)

        # Save to a CSV file
        df = pd.DataFrame({
            'True Values': true_values.numpy()[:, 0].flatten(),
            'Predictions': predictions.numpy()[:, 0].flatten()
        })
        # Add input features to the dataframe
        for i in range(inputs_all.shape[1]):
            df[f'Input Feature {i+1}'] = inputs_all[:, i].flatten()
        
        df.to_csv(path, index=False)
        logging.info(f"True values, predictions, and inputs saved to {path}")
    
    def evaluation_figure(self, loader='val', ax_reduce=0.5, n_bins=50):

        if loader == 'val':
            true_values, predictions = self.ytrue_ypred(self.val_loader)
        elif loader == 'train':
            true_values, predictions = self.ytrue_ypred(self.train_loader)
        
        # Select only the first label for eval
        if self.n_labels > 1:
            true_values = true_values[:, 0].unsqueeze(1)
            predictions = predictions[:, 0].unsqueeze(1)

        # TODO: enable test loader
        # elif loader == 'test':
        #     true_values, predictions = self.ytrue_ypred(self.test_loader)

        # Unscale the true values and predictions
        # true_values = self.unscale_ytrue_ypred(true_values)
        # predictions = self.unscale_ytrue_ypred(predictions)

        # Fit a linear regression model on the predictions and true values, then plot the trendline
        reg = LinearRegression().fit(true_values, predictions)
        gradient = reg.coef_[0][0]

        # Make an evaluation figure, with a hexbin plot and a histogram of the true and predicted
        plt.figure(figsize=(8,12), dpi=300)
        plt.subplot(2, 1, 1)
        plt.hexbin(true_values, predictions, gridsize=150, cmap='Blues', mincnt=10, bins='log')
        plt.colorbar(label='Counts')
        # plt.scatter(true_values, predictions, s=0.5, alpha=0.2)
        plt.plot([-5, 5], [-5, 5], 'r--', label='1:1')
        plt.plot(true_values, reg.predict(true_values), 'r', label='Linear fit: '+ str(round(gradient,2)))
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        axmin = min(true_values.min(), predictions.min())
        axmax = max(true_values.max(), predictions.max())
        plt.ylim(axmin*ax_reduce, axmax*ax_reduce)
        plt.xlim(axmin*ax_reduce, axmax*ax_reduce)

        mse = torch.nn.functional.mse_loss(predictions, true_values)
        persistence_mse = torch.nn.functional.mse_loss(torch.zeros_like(true_values), true_values)
        rmse_cms = torch.sqrt(mse) * 100  # Convert to cm/s
        skill = 1 - (mse / persistence_mse)

        plt.title(f'RMSE: {rmse_cms:.2e} cm/s, Skill: {skill:.3f}, Gradient: {gradient:.4f}')
        plt.legend()

        plt.subplot(2, 1, 2)
        bin_edges = np.linspace(axmin, axmax, n_bins + 1)
        plt.hist(true_values, bins=bin_edges, alpha=0.5, label='True Values', color='blue', density=True)
        plt.hist(predictions, bins=bin_edges, alpha=0.5, label='Predictions', color='orange', density=True)
        plt.xlabel('Values')
        plt.ylabel('Counts')
        plt.title('Normalised histogram of True Values and Predictions')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.arguments['results_path'] + f'evaluation_{loader}.png')

        logging.info(f"Evaluation figure for {loader} set saved.")

    def plot_north_polar_map(self, loader='val', stride_base=800_000):
        if loader == 'val':
            true_values, predictions = self.ytrue_ypred(self.val_loader)
            indices = self.val_loader.dataset.indices
        elif loader == 'train':
            true_values, predictions = self.ytrue_ypred(self.train_loader)
            indices = self.train_loader.dataset.indices
        
        # Select only the first label for eval
        if self.n_labels > 1:
            true_values = true_values[:, 0].unsqueeze(1)
            predictions = predictions[:, 0].unsqueeze(1)
        
        da = self.data_manager.raw_data.isel(z=indices)

        stride = max(1, true_values.shape[0] // stride_base)
        logging.info(f"Using stride of {stride} for sampling")

        vals = torch.nn.functional.l1_loss(predictions, true_values, reduction='none').numpy()[::stride]
        lats = da.coords['lat'].values[::stride]
        lons = da.coords['lon'].values[::stride]

        fig = plt.figure(figsize=(10, 8))

        projection = ccrs.NorthPolarStereo()
        ax = plt.axes(projection=projection)
        ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, zorder=1, facecolor='gray')
        ax.coastlines(resolution='110m', zorder=2)
        ax.gridlines()

        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        hb = ax.hexbin(
            lons, lats,
            C=vals,
            gridsize=500,
            cmap='viridis',
            transform=ccrs.PlateCarree(),
            reduce_C_function=np.mean,
            mincnt=1
        )

        cbar = plt.colorbar(hb, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
        cbar.set_label('Sea Ice Velocity MAE')

        plt.title("Sea Ice Velocity MAE Map")
        fig.savefig(self.arguments['results_path'] + f'map_{loader}.png', dpi=300)

        logging.info(f"MAE map for {loader} set saved.")
    
    def save_model(self, path):

        model_recreator = {
            'state_dict': self.model.state_dict(),
            'architecture': self.architecture,
            'n_features': self.n_features,
            'n_labels': self.n_labels,
            'scaler': self.scaler
        }

        with open(path, 'wb') as f:
            pickle.dump(model_recreator, f)                 
        

def train_save_eval(arguments):

    nn_capsule = NNCapsule(arguments)

    nn_capsule.train()

    nn_capsule.plot_train_losses(nn_capsule.train_losses, nn_capsule.val_losses)
    nn_capsule.evaluation_figure('train')
    nn_capsule.plot_north_polar_map('train')

    nn_capsule.save_model(arguments['results_path'] + 'nn_model_recreator.pkl')

    if arguments['save_data']:
        nn_capsule.data_manager.save_datasets(arguments['results_path'] + 'data_splits/')

    if arguments['save_val']:
        nn_capsule.save_ytrue_ypred_inputs(nn_capsule.val_loader, arguments['results_path'] + 'ytrue_ypred_val.csv')

    logging.info("Training complete. Results saved in: " + arguments['results_path'])
    
