import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import FunctionTransformer, Pipeline


class InvertableColumnTransformer(ColumnTransformer):
    """
    From https://github.com/scikit-learn/scikit-learn/issues/11463#issuecomment-1674435238

    Adds an inverse transform method to the standard sklearn.compose.ColumnTransformer.

    Warning this is flaky and use at your own risk.  Validation checks that the column count in
    `transformers` are in your object `X` to be inverted.  Reordering of columns will break things!
    """

    def inverse_transform(self, X):
        arrays = []
        for name, indices in self.output_indices_.items():
            transformer = self.named_transformers_.get(name, None)
            arr = X[:, indices.start : indices.stop]

            if transformer in (None, "passthrough", "drop"):
                pass

            else:
                arr = transformer.inverse_transform(arr)

            arrays.append(arr)

        retarr = np.concatenate(arrays, axis=1)

        if retarr.shape[1] != X.shape[1]:
            raise ValueError(
                f"Received {X.shape[1]} columns but transformer expected {retarr.shape[1]}"
            )

        return retarr


class LogFLTransformer:
    def __init__(
        self,
        features,
        labels,
        feature_names,
        feature_scale_factor: float,
        label_scale_factor: float,
    ):
        self.feature_scale_factor = feature_scale_factor
        self.label_scale_factor = label_scale_factor

        self.feature_vel_scaler = Pipeline(
            [
                (
                    "log",
                    FunctionTransformer(
                        func=self._log_scaled,
                        inverse_func=self._exp_scaled,
                        validate=True,
                        kw_args={"scale_factor": self.feature_scale_factor},
                        inv_kw_args={"scale_factor": self.feature_scale_factor},
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )

        # Find which velocity values are present
        vel_indices = [
            feature_names.index(n) for n in ["sivelv", "sivelu"] if n in feature_names
        ]

        # Define a scaler for all features
        self.feature_scaler = InvertableColumnTransformer(
            transformers=[("velocity", self.feature_vel_scaler, vel_indices)],
            remainder=StandardScaler(),
        )

        self.label_scaler = Pipeline(
            [
                (
                    "log",
                    FunctionTransformer(
                        func=self._log_scaled,
                        inverse_func=self._exp_scaled,
                        validate=True,
                        kw_args={"scale_factor": self.label_scale_factor},
                        inv_kw_args={"scale_factor": self.label_scale_factor},
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )

        self.feature_scaler.fit(features)
        self.label_scaler.fit(labels)

    def _log_scaled(self, x, scale_factor=1):
        return np.sign(x) * np.log1p(np.abs(x) * scale_factor)

    def _exp_scaled(self, x, scale_factor=1):
        return np.sign(x) * (np.expm1(np.abs(x)) / scale_factor)
