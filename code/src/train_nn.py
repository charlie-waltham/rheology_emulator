import logging
import pickle

import torch
import pandas as pd
import matplotlib.pyplot as plt

from . import utils

def setup_logging(log_file='train_nn.log'):
    """
    Set up logging to a file.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging setup complete.")

class NNCapsule:
    def __init__(self, arguments):
        self.arguments = arguments

        # Load data
        self.zarr_fmt = arguments['zarr_fmt']
        if self.zarr_fmt == 'fmt1':
            from .data_managers.fmt1 import TorchDataManager
            self.data_manager = TorchDataManager(arguments['pairs_path'], arguments, zarr_fmt=self.zarr_fmt, difference_labels=arguments['difference_labels'])
            self.train_loader = self.data_manager.train.dataset
            self.val_loader = self.data_manager.val.dataset
            self.n_features = self.train_loader[0][0].shape[1]
            self.n_labels = self.train_loader[0][1].shape[1]
            self.n_batches = len(self.train_loader)
            self.n_samples = len(self.train_loader.dataset)
            self.n_observations = self.train_loader[10][0].shape[0]
        elif self.zarr_fmt == 'fmt2':
            from .data_managers.fmt2 import TorchDataManager
            self.data_manager = TorchDataManager(arguments['pairs_path'], arguments, difference_labels=arguments['difference_labels'])
            self.n_features = self.data_manager.n_features
            self.n_labels = self.data_manager.n_labels
            self.n_samples = self.data_manager.n_train
            self.n_batches = self.data_manager.n_batches_train
            self.train_loader = self.data_manager.train_loader
            self.val_loader = self.data_manager.val_loader
            self.test_loader = self.data_manager.test_loader

        if self.data_manager.scale:
            self.scaler = self.data_manager.scaler
            self.label_scaler = self.data_manager.label_vel_scaler
        else:
            self.scaler = None
            self.label_scaler = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'Using device: {self.device}')

        # Define model
        self.architecture = arguments['architecture']
        self.parameters = arguments['parameters']
        self.model = utils.define_nn(self.architecture, self.n_features, self.n_labels, self.device)
        # TODO: split the below up so that they're called separately, or do some order agnostic unpacking of all the parameters
        self.criterion, self.optimizer, self.n_epochs = utils.nn_options(self.model, self.parameters)
        self.train_losses = []
        self.val_losses = []

        # Set up logging
        setup_logging(log_file='train_nn.log')
        self._log_summary()

    def _log_summary(self):
        logging.info("Model Summary:")
        logging.info(f"Architecture: {self.architecture}")
        logging.info(f"Parameters: {self.parameters}")
        logging.info(f"Number of training samples: {self.n_samples}")
        logging.info(f"Number of batches: {self.n_batches}")
        logging.info(f"Number of features: {self.n_features}")
        logging.info(f"Number of labels: {self.n_labels}")

    def train(self):
        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                with torch.amp.autocast(self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.detach()
            self.train_losses.append(running_loss.item() / len(self.train_loader))

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            self.val_losses.append(val_loss / len(self.val_loader))

            logging.info(f"Epoch {epoch+1}, Train Loss: {self.train_losses[-1]:.2e}, Val Loss: {self.val_losses[-1]:.2e}")

        logging.info("Training complete.")
        logging.info(f"Final Train Loss: {self.train_losses[-1]:.2e}, Final Val Loss: {self.val_losses[-1]:.2e}")

    def plot_train_losses(self, train_losses, val_losses):
        fig = plt.figure(figsize=(5, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.arguments['results_path'] + 'train_losses.png')
        logging.info("Training losses plotted and saved.")
        return fig
    
    def ytrue_ypred(self, loader):
        predictions = []
        true_values = []
        with torch.no_grad():
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
        with torch.no_grad():
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

        indices = loader.dataset.indices

        # Unscale the true values, predictions and inputs
        if (self.data_manager.scale):
            predictions = torch.tensor(self.label_scaler.inverse_transform(predictions))
            true_values = torch.tensor(self.label_scaler.inverse_transform(true_values))
            inputs_all = self.scaler.inverse_transform(inputs_all)

        # Save to a CSV file
        df = pd.DataFrame({
            'true_sivelu': true_values.numpy()[:, 0].flatten(),
            'true_sivelv': true_values.numpy()[:, 1].flatten(),
            'pred_sivelu': predictions.numpy()[:, 0].flatten(),
            'pred_sivelv': predictions.numpy()[:, 1].flatten(),
            'indices': indices
        })
        # Add input features to the dataframe
        for i in range(inputs_all.shape[1]):
            df[f'feature_{i+1}'] = inputs_all[:, i].flatten()
        
        df.to_csv(path, index=False)
        logging.info(f"True values, predictions, and inputs saved to {path}")
    
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
    #nn_capsule.save_model(arguments['results_path'] + 'nn_model_recreator.pkl')

    if arguments['save_data']:
        nn_capsule.data_manager.save_datasets(arguments['results_path'] + 'data_splits/')

    if arguments['save_val']:
        nn_capsule.save_ytrue_ypred_inputs(nn_capsule.val_loader, arguments['results_path'] + 'ytrue_ypred_val.csv')

    logging.info("Training complete. Results saved in: " + arguments['results_path'])
