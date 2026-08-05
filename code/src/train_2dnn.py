import logging

import torch
from matplotlib import pyplot as plt

from . import utils
from .data_managers.two_dim import TorchDataManager


class NNCapsule:
    def __init__(self, arguments):
        self.arguments = arguments
        self.data_manager = TorchDataManager(arguments)

        self.n_features = self.data_manager.n_features
        self.n_labels = self.data_manager.n_labels
        self.n_samples = self.data_manager.n_train
        self.n_batches = self.data_manager.n_batches_train
        self.train_loader = self.data_manager.train_loader
        self.val_loader = self.data_manager.val_loader
        self.test_loader = self.data_manager.test_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Define model
        self.architecture = arguments["architecture"]
        self.parameters = arguments["parameters"]
        self.model = utils.define_nn(
            self.architecture, self.n_features, self.n_labels, self.device
        )

        self.criterion, self.optimizer, self.n_epochs = utils.nn_options(
            self.model, self.parameters
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=3
        )

        self.train_losses = []
        self.val_losses = []

        if not self.data_manager.difference_labels:
            self.baseline_indices = [
                self.data_manager.train_features.index(label)
                for label in self.data_manager.train_labels
            ]

        self._log_summary()
    
    def _log_summary(self):
        logging.info("Model Summary:")
        logging.info(f"Architecture: {self.architecture}")
        logging.info(f"Parameters: {self.parameters}")
        logging.info(f"Number of training samples: {self.n_samples}")
        logging.info(f"Number of batches: {self.n_batches}")
        logging.info(f"Number of features: {self.n_features}")
        logging.info(f"Number of labels: {self.n_labels}")
        num_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Number of parameters: {num_params}")

    def _prepare_batch(self, batch):
        inputs, targets, mask = batch
        return (
            inputs.to(self.device, non_blocking=True),
            targets.to(self.device, non_blocking=True),
            mask.to(self.device, non_blocking=True),
        )

    def _get_baseline(self, inputs, outputs):
        if self.data_manager.difference_labels:
            return torch.zeros_like(outputs)
        return inputs[:, self.baseline_indices]

    def _compute_losses(self, inputs, outputs, targets, mask):
        loss = self.criterion(outputs, targets, mask.unsqueeze(1))
        baseline = self._get_baseline(inputs, outputs)
        bloss = self.criterion(baseline, targets, mask.unsqueeze(1))
        return loss, bloss

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        baseline_loss = 0.0

        for batch in self.train_loader:
            inputs, targets, mask = self._prepare_batch(batch)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss, bloss = self._compute_losses(inputs, outputs, targets, mask)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach()
            baseline_loss += bloss.detach()

        epoch_loss = running_loss.item() / len(self.train_loader)
        avg_baseline_loss = (baseline_loss / len(self.train_loader)).item()
        skill = 1 - epoch_loss / avg_baseline_loss
        return epoch_loss, skill

    def _validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        baseline_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets, mask = self._prepare_batch(batch)
                outputs = self.model(inputs)

                loss, bloss = self._compute_losses(inputs, outputs, targets, mask)
                running_loss += loss.detach()
                baseline_loss += bloss.detach()

        epoch_loss = running_loss.item() / len(self.val_loader)
        avg_baseline_loss = (baseline_loss / len(self.val_loader)).item()
        skill = 1 - epoch_loss / avg_baseline_loss
        return epoch_loss, skill

    def train(self):
        best_loss = torch.inf

        for epoch in range(self.n_epochs):
            train_loss, train_skill = self._train_epoch()
            self.train_losses.append(train_loss)

            val_loss, val_skill = self._validate_epoch()
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            best_str = " (best)" if val_loss < best_loss else ""
            logging.info(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.2e}, Train Skill: {train_skill:.2f}, Val Loss: {val_loss:.2e}{best_str}, Val Skill: {val_skill:.2f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.arguments["results_path"] + "model.pkl")

        logging.info("Training complete.")

    def plot_train_losses(self, train_losses, val_losses):
        fig = plt.figure(figsize=(5, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.arguments["results_path"] + "train_losses.png")
        logging.info("Training losses plotted and saved.")
        return fig

def train_save_eval(arguments):
    nn_capsule = NNCapsule(arguments)

    if arguments["save_data"]:
        nn_capsule.data_manager.save_datasets(
            arguments["results_path"] + "data_splits/"
        )

    nn_capsule.train()
    nn_capsule.plot_train_losses(nn_capsule.train_losses, nn_capsule.val_losses)

    logging.info("Training complete. Results saved in: " + arguments["results_path"])
