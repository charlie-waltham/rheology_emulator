from pathlib import Path

import torch
import xarray as xr
import yaml
from scaling import LogFLTransformer
from utils import define_nn


class RheologyModel:
    """
    A wrapper class for loading and running inference with a trained rheology neural network model.

    This class handles loading the model architecture and weights, setting up data scaling
    transformers based on training configurations, and executing forward passes.

    Args:
        model_dir (Path): The directory containing the trained model artifacts, including
            'model.pkl' and 'used_training_config.yaml'.
        code_dir (Path): The directory containing the source code, used to locate
            architecture definitions referenced in the config.
        sample_data (xr.Dataset): A sample xarray Dataset containing 'features' and 'labels'
            (and optionally 'd_labels'). This is used to fit or initialize the
            LogFLTransformer for scaling if feature scaling was used during training.
        device (str, optional): The device to run the model on (e.g., "cpu", "cuda").
            Defaults to "cpu".
    """

    def __init__(
        self,
        model_dir: Path,
        code_dir: Path,
        sample_data: xr.Dataset,
        device: str = "cpu",
    ):
        self.model_dir = model_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")
        self.device = device

        with open(model_dir / "used_training_config.yaml") as file:
            self.config: dict = yaml.safe_load(file)

        if self.config.get("scale_features"):
            if self.config.get("difference_labels"):
                if sample_data.get("d_labels") is None:
                    d_sivelu = (
                        sample_data.labels.loc["sivelu"]
                        - sample_data.features.loc["sivelu"]
                    )
                    d_sivelv = (
                        sample_data.labels.loc["sivelv"]
                        - sample_data.features.loc["sivelv"]
                    )
                    sample_data = sample_data.assign(
                        {"d_labels": xr.concat([d_sivelv, d_sivelu], dim="d_label")}
                    )
                    sample_data = sample_data.assign_coords(
                        {"d_label": ["d_sivelv", "d_sivelu"]}
                    )

                labels = sample_data.d_labels.values
            else:
                labels = sample_data.labels.values

            self.scaler = LogFLTransformer(
                sample_data.features.values.T,
                labels.T,
                self.config.get("train_features"),
                20,
                100,
            )

        self.model: torch.nn.Module = define_nn(
            code_dir / self.config.get("architecture"),
            len(self.config.get("train_features")),
            len(self.config.get("train_labels")),
            self.device,
        )
        self.model.load_state_dict(torch.load(model_dir / "model.pkl"))
        self.model.eval()

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.get("scale_features"):
            features = self.scaler.feature_scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(features).to("cpu")

        if self.config.get("scale_features"):
            outputs = self.scaler.label_scaler.inverse_transform(outputs)

        return torch.tensor(outputs, dtype=torch.float32)
