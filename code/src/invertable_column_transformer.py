import numpy as np
from sklearn.compose import ColumnTransformer

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
            arr = X[:, indices.start: indices.stop]

            if transformer in (None, "passthrough", "drop"):
                pass

            else:
                arr = transformer.inverse_transform(arr)

            arrays.append(arr)

        retarr = np.concatenate(arrays, axis=1)

        if retarr.shape[1] != X.shape[1]:
            raise ValueError(f"Received {X.shape[1]} columns but transformer expected {retarr.shape[1]}")

        return retarr