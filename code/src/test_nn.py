import logging
from pathlib import Path
import pickle

import torch
import pandas as pd
from captum.attr import DeepLiftShap

from .data_managers.fmt2 import TorchDataManager
from . import utils


class NNCapsule:
    def __init__(self, arguments):
        self.arguments = arguments

        self.architecture = arguments["architecture"]
        self.parameters = arguments["parameters"]
        self.data_manager = TorchDataManager(arguments)

        self.test_loader = self.data_manager.test_loader
        self.scaler = self.data_manager.scaler if self.data_manager.scale else None
        self.n_features = self.data_manager.n_features
        self.n_labels = self.data_manager.n_labels
        self.n_samples = self.data_manager.n_test

        model_path = Path(arguments["eval_path"]) / "model.pkl"
        if not model_path.exists():
            logging.error(f"Model not found at {model_path}")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.model = utils.define_nn(
            self.architecture, self.n_features, self.n_labels, self.device
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        self.model.to(self.device)

        self.criterion, self.optimizer, self.n_epochs = utils.nn_options(
            self.model, self.parameters
        )

        self._log_summary()

    def _log_summary(self):
        logging.info("Model Summary:")
        logging.info(f"Architecture: {self.architecture}")
        logging.info(f"Number of samples: {self.n_samples}")
        logging.info(f"Number of features: {self.n_features}")
        logging.info(f"Number of labels: {self.n_labels}")

    def test(self):
        self.model.eval()
        losses = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = (
                    inputs.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                )
                outputs = self.model(inputs)
                losses = self.criterion(outputs, targets)
                losses += losses.detach()
        self.loss = losses.item() / len(self.test_loader)

        logging.info("Testing complete.")
        logging.info(f"Loss: {self.loss:.2e}")

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
        if self.data_manager.scale:
            predictions = self.scaler.label_scaler.inverse_transform(predictions)
            true_values = self.scaler.label_scaler.inverse_transform(true_values)

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
        if self.data_manager.scale:
            predictions = torch.tensor(
                self.scaler.label_scaler.inverse_transform(predictions)
            )
            true_values = torch.tensor(
                self.scaler.label_scaler.inverse_transform(true_values)
            )
            inputs_all = self.scaler.feature_scaler.inverse_transform(inputs_all)

        # Save to a CSV file
        df = pd.DataFrame(
            {
                "true_sivelu": true_values.numpy()[:, 1].flatten(),
                "true_sivelv": true_values.numpy()[:, 0].flatten(),
                "pred_sivelu": predictions.numpy()[:, 1].flatten(),
                "pred_sivelv": predictions.numpy()[:, 0].flatten(),
                "indices": indices,
            }
        )
        # Add input features to the dataframe
        for i in range(inputs_all.shape[1]):
            df[f"feature_{i + 1}"] = inputs_all[:, i].flatten()

        df.to_csv(path, index=False)
        logging.info(f"True values, predictions, and inputs saved to {path}")

    def save_attributions(self, path, n_baseline=100, n_samples=1000, batch_size=50):
        features = self.data_manager.dataset.features

        baseline_indices = torch.randperm(len(features))[:n_baseline]
        indices = torch.randperm(len(features))[:n_samples]
        baseline_features = features[baseline_indices].to(self.device)
        attr_features = features[indices].to(self.device)

        self.model.eval()
        explainer = DeepLiftShap(self.model)

        results = {}
        for target_label in range(self.n_labels):
            attributions_list = []
            for i in range(0, len(attr_features), batch_size):
                logging.info(f"Processing batch {i // batch_size + 1}/{len(attr_features) // batch_size} for label {target_label}")
                batch_attr = explainer.attribute(
                    attr_features[i : i + batch_size], baseline_features, target=target_label
                )
                attributions_list.append(batch_attr.cpu().detach())

            attributions = torch.cat(attributions_list, dim=0)
            results[target_label] = attributions.numpy()

        pickle.dump(results, open(path, "wb"))
        logging.info(f"Attributions saved to {path}")


def test_save_eval(arguments):
    nn_capsule = NNCapsule(arguments)
    nn_capsule.test()

    nn_capsule.save_ytrue_ypred_inputs(
        nn_capsule.test_loader, arguments["eval_path"] + "/ytrue_ypred_test.csv"
    )

    nn_capsule.save_attributions(arguments["eval_path"] + "/attributions.pkl")
