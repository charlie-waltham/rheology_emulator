import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append("src")

import matplotlib.pyplot as plt
import nc_time_axis  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from inference import RheologyModel
from scipy.spatial import KDTree

plt.style.use("seaborn-v0_8")

t = time.time()
dataset: xr.Dataset = xr.open_zarr("../data/processed/antarctic/all/1.zarr")
dataset = dataset.sel(
    {
        "feature": [
            "siconc",
            "sithic",
            "sivelv",
            "sivelu",
            "utau_ai",
            "utau_oi",
            "vtau_ai",
            "vtau_oi",
        ]
    }
).load()
print(f"{time.time() - t:.2f}s to load data")

t = time.time()
time_groups = list(dataset.groupby("time_features"))
print(f"{time.time() - t:.2f}s to group by time")

t = time.time()
# data_subset = dataset.isel(z=slice(None, None, 1000))
model = RheologyModel(
    Path("../results/nn/antarctic-tests/all/all/20260225_0952"),
    Path("../code"),
    dataset,
    "cuda",
)
# data_subset.close()
print(f"{time.time() - t:.2f}s to import model")

previous_vel = None
previous_coords = None
initial_vel = None
unmatched_points = 0

metrics = {
    "time": [],
    "mae": [],
    "skill": [],
    "mean": [],
    "std": [],
    "min": [],
    "max": [],
}
print(f"Found {len(time_groups)} unique timesteps")

for i, (current_time, data) in enumerate(time_groups):
    t = time.time()

    current_coords = np.column_stack((data["lat"].values, data["lon"].values))

    if previous_vel is not None and previous_coords is not None:
        # Use a KDTree to correlate points from t-1 and t
        tree = KDTree(previous_coords)
        _, indices = tree.query(current_coords, distance_upper_bound=1e-5, workers=-1)
        valid_matches = indices < len(previous_coords)
        valid_positions = np.nonzero(valid_matches)[0]

        data = data.isel(z=valid_positions)
        current_coords = current_coords[valid_positions]

        matched_prev_indices = indices[valid_matches]
        initial_vel = initial_vel[:, matched_prev_indices]

        data.features.loc["sivelv"] = previous_vel[0, matched_prev_indices].numpy()
        data.features.loc["sivelu"] = previous_vel[1, matched_prev_indices].numpy()
    else:
        # First iteration, fetch current velocity features
        initial_vel = torch.tensor(data.features.loc[["sivelv", "sivelu"]].values)

    # Output is a residual (t+1 - t)
    output = model(torch.tensor(data.features.values.T)).T
    current_vel = torch.tensor(data.features.loc[["sivelv", "sivelu"]].values) + output
    true_current_vel = torch.tensor(data.labels.loc[["sivelv", "sivelu"]].values)

    # Calculate vectors: Total change since t=0
    pred_cumulative_residual = current_vel - initial_vel
    true_cumulative_residual = true_current_vel - initial_vel

    error_mag = torch.linalg.vector_norm(
        pred_cumulative_residual - true_cumulative_residual, dim=0
    )
    # Baseline error = 0 - true_cumulative_residual = -true_cumulative_residual
    baseline_error_mag = torch.linalg.vector_norm(-true_cumulative_residual, dim=0)

    current_mse = F.mse_loss(pred_cumulative_residual, true_cumulative_residual).item()
    baseline_mse = F.mse_loss(
        torch.zeros_like(true_cumulative_residual), true_cumulative_residual
    ).item()
    pred_mag = torch.linalg.vector_norm(pred_cumulative_residual, dim=0)

    metrics["time"].append(current_time)
    metrics["mae"].append(error_mag.mean().item())
    metrics["skill"].append((1 - current_mse / baseline_mse) if baseline_mse > 0 else 0)
    metrics["mean"].append(pred_mag.mean().item())
    metrics["std"].append(pred_mag.std().item())
    metrics["min"].append(pred_mag.min().item())
    metrics["max"].append(pred_mag.max().item())

    print(
        f"\n-----------------------------------\n{i + 1}/{len(time_groups)} ({time.time() - t:.2f}s):"
    )
    for key in metrics:
        print(f"{key}: {metrics[key][-1]}")

    previous_vel = current_vel
    previous_coords = current_coords


fig, axs = plt.subplots(2, 3, figsize=(9, 6))
axs = axs.flatten()
formatter = nc_time_axis.CFTimeFormatter("%H:%M", "noleap")
plt.suptitle("Autoregression Performance Metrics")

titles = [
    "Mean Absolute Error",
    "Skill",
    "Mean",
    "Std. Deviation",
    "Min",
    "Max",
]
for i, key in enumerate(metrics):
    if key == "time":
        continue
    axs[i - 1].plot(metrics["time"], metrics[key])
    axs[i - 1].set_title(titles[i - 1])
    axs[i - 1].xaxis.set_major_formatter(formatter)
fig.tight_layout()

results_path = Path("../results/autoregression")
results_path.mkdir(parents=True, exist_ok=True)
plt.savefig(results_path / (datetime.now().strftime("%Y%m%d_%H%M") + ".png"), dpi=300)
plt.close()
