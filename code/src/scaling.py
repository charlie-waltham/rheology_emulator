import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler


class InvertableColumnTransformer(ColumnTransformer):
    """
    From https://github.com/scikit-learn/scikit-learn/issues/11463#issuecomment-1674435238

    Adds an inverse transform method to the standard sklearn.compose.ColumnTransformer.

    Warning this is flaky and use at your own risk.  Validation checks that the column count in
    `transformers` are in your object `X` to be inverted.  Reordering of columns will break things!
    """

    def inverse_transform(self, X):
        arrays = []
        original_col_indices = []
        for name, indices in self.output_indices_.items():
            transformer = self.named_transformers_.get(name, None)
            arr = X[:, indices.start : indices.stop]

            if (
                transformer in (None, "passthrough", "drop")
                or indices.start == indices.stop
            ):
                pass

            else:
                arr = transformer.inverse_transform(arr)

            arrays.append(arr)

            for trans_name, trans_obj, orig_cols in self.transformers_:
                if trans_name == name:
                    if trans_name == "remainder":
                        all_specified = [
                            col
                            for t_name, _, t_cols in self.transformers_
                            if t_name != "remainder" and t_name != "drop"
                            for col in t_cols
                        ]
                        orig_cols = [
                            c for c in range(X.shape[1]) if c not in all_specified
                        ]
                    original_col_indices.extend(orig_cols)
                    break

        retarr = np.concatenate(arrays, axis=1)

        if retarr.shape[1] != X.shape[1]:
            raise ValueError(
                f"Received {X.shape[1]} columns but transformer expected {retarr.shape[1]}"
            )

        if len(original_col_indices) == retarr.shape[1]:
            rev_perm = np.argsort(original_col_indices)
            retarr = retarr[:, rev_perm]

        return retarr


class LogFLTransformer:
    def __init__(
        self,
        features,
        labels,
        feature_names,
        label_names,
        features_to_log_scale,
        labels_to_log_scale,
        feature_scale_factor: float,
        label_scale_factor: float,
    ):
        self.feature_scale_factor = feature_scale_factor
        self.label_scale_factor = label_scale_factor

        self.feature_log_scaler = Pipeline(
            [
                (
                    "log",
                    FunctionTransformer(
                        func=self._log_scaled,
                        inverse_func=self._exp_scaled,
                        validate=True,
                        check_inverse=False,
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
            transformers=[("velocity", self.feature_log_scaler, vel_indices)],
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
                        check_inverse=False,
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
