import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from captum.attr import DeepLiftShap

from . import utils
from .data_managers.two_dim import TorchDataManager


class NNCapsule:
    def __init__(self, arguments):
        self.arguments = arguments

        self.architecture = arguments["architecture"]
        self.parameters = arguments["parameters"]
        self.data_manager = TorchDataManager(arguments)

        self.test_loader = self.data_manager.test_loader

        self.n_features = self.data_manager.n_features
        self.n_labels = self.data_manager.n_labels

        self.inputs = []
        self.predictions = []
        self.true_values = []

        model_path = Path(arguments["eval_path"]) / "model.pkl"
        if not model_path.exists():
            logging.error(f"Model not found at {model_path}")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.model = utils.define_nn(
            self.architecture, self.n_features, self.n_labels, self.device
        )

        self.criterion, self.optimizer, self.n_epochs = utils.nn_options(
            self.model, self.parameters
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        self.model.to(self.device)

        if not self.data_manager.difference_labels:
            self.baseline_indices = [
                self.data_manager.train_features.index(label)
                for label in self.data_manager.train_labels
            ]

        self._log_summary()

    def _log_summary(self):
        logging.info("Model Summary:")
        logging.info(f"Architecture: {self.architecture}")
        logging.info(f"Number of features: {self.n_features}")
        logging.info(f"Number of labels: {self.n_labels}")
        num_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Number of parameters: {num_params}")

    def test(self):
        self.model.eval()
        running_loss = 0

        with torch.no_grad():
            for inputs, targets, mask in self.test_loader:
                inputs, targets, mask = (
                    inputs.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                    mask.to(self.device, non_blocking=True),
                )
                outputs = self.model(inputs)

                running_loss += self.criterion(outputs, targets, mask.unsqueeze(1)).detach()

                self.inputs.append(inputs)
                self.predictions.append(outputs)
                self.true_values.append(targets)

        self.loss = running_loss.item() / len(self.test_loader)
        self.inputs = torch.cat(self.inputs, dim=0).to("cpu")
        self.predictions = torch.cat(self.predictions, dim=0).to("cpu")
        self.true_values = torch.cat(self.true_values, dim=0).to("cpu")

        logging.info("Testing complete.")
        logging.info(f"Loss: {self.loss:.2e}")

    def save_ytrue_ypred_inputs(self, loader, path):
        indices = loader.dataset.indices

        # Permute from (N, C, H, W) to (N, H, W, C) so channels/labels are the last dimension
        inputs = self.inputs.permute(0, 2, 3, 1)
        predictions = self.predictions.permute(0, 2, 3, 1)
        true_values = self.true_values.permute(0, 2, 3, 1)

        # Unscale the true values and predictions
        if self.data_manager.scaling is not None:
            input_shape = inputs.shape
            input_flat_size = np.multiply.reduce(input_shape[:3])

            out_shape = predictions.shape
            out_flat_size = np.multiply.reduce(out_shape[:3])

            # Convert 2D flattened tensors to numpy arrays for scikit-learn transformers
            flat_inputs = inputs.reshape(input_flat_size, self.n_features).detach().cpu().numpy()
            flat_pred = predictions.reshape(out_flat_size, self.n_labels).detach().cpu().numpy()
            flat_true = true_values.reshape(out_flat_size, self.n_labels).detach().cpu().numpy()

            # Apply inverse_transform via data_manager's label_scaler
            unscaled_inputs = self.data_manager.feature_scaler.inverse_transform(flat_inputs)
            unscaled_pred = self.data_manager.label_scaler.inverse_transform(flat_pred)
            unscaled_true = self.data_manager.label_scaler.inverse_transform(flat_true)

            inputs = torch.tensor(unscaled_inputs.reshape(input_shape))
            predictions = torch.tensor(unscaled_pred.reshape(out_shape))
            true_values = torch.tensor(unscaled_true.reshape(out_shape))
        else:
            inputs = inputs.detach().cpu()
            predictions = predictions.detach().cpu()
            true_values = true_values.detach().cpu()

        inputs_sivel = inputs[..., self.baseline_indices]

        inputs_sivel_vector = inputs_sivel[..., 0:1] * inputs_sivel[..., 1:]
        predictions_vector = predictions[..., 0:1] * predictions[..., 1:]
        true_values_vector = true_values[..., 0:1] * true_values[..., 1:]

        # Apply spatial mask to evaluate only valid ocean grid points
        mask_tensor = torch.tensor(self.data_manager.mask, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
        mask_expanded = mask_tensor.expand_as(predictions_vector)

        pred_ocean = predictions_vector[mask_expanded]
        true_ocean = true_values_vector[mask_expanded]
        base_ocean = inputs_sivel_vector[mask_expanded]

        mse = F.mse_loss(pred_ocean, true_ocean)
        baseline_mse = F.mse_loss(base_ocean, true_ocean)
        skill = 1 - mse / baseline_mse

        logging.info(f"MSE: {mse:.2e}")
        logging.info(f"Baseline MSE: {baseline_mse:.2e}")
        logging.info(f"Skill: {skill:.2f}")

        # Save to netCDF
        ds = xr.Dataset(
            data_vars={
                "pred": (
                    ("indices", "x", "y", "directions"),
                    predictions_vector.numpy(),
                ),
                "true": (
                    ("indices", "x", "y", "directions"),
                    true_values_vector.numpy(),
                ),
            },
            coords={"indices": indices, "directions": ["u", "v"]},
        )
        logging.info(ds)
        ds.to_netcdf(path)

        logging.info(f"True values and predictions saved to {path}")

    def _get_baseline(self, inputs, outputs):
            if self.data_manager.difference_labels:
                return torch.zeros_like(outputs)
            return inputs[:, self.baseline_indices]

    def save_attributions(self, path):
        features = self.data_manager.dataset.features

        baseline_indices = torch.randperm(len(features))[
            : self.arguments["attr_baseline"]
        ]
        indices = torch.randperm(len(features))[: self.arguments["attr_samples"]]
        baseline_features = features[baseline_indices].to(self.device)
        attr_features = features[indices].to(self.device)

        self.model.eval()
        explainer = DeepLiftShap(self.model)

        results = {}
        for target_label in range(self.n_labels):
            logging.info(f"Processing attributions for label {target_label}")
            attributions_list = []
            for i in range(0, len(attr_features), self.arguments["attr_batch_size"]):
                batch_attr = explainer.attribute(
                    attr_features[i : i + self.arguments["attr_batch_size"]],
                    baseline_features,
                    target=target_label,
                )
                attributions_list.append(batch_attr.cpu().detach())

            attributions = torch.cat(attributions_list, dim=0)
            results[target_label] = attributions.numpy()

        with open(path, "wb") as file:
            pickle.dump(results, file)
        logging.info(f"Attributions saved to {path}")


def test_save_eval(arguments):
    nn_capsule = NNCapsule(arguments)
    nn_capsule.test()

    nn_capsule.save_ytrue_ypred_inputs(
        nn_capsule.test_loader, arguments["eval_path"] + "/ytrue_ypred_test.nc"
    )

    #nn_capsule.save_attributions(arguments["eval_path"] + "/attributions.pkl")
