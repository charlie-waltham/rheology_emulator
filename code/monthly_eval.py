import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, MonthLocator

plt.style.use("seaborn-v0_8")

runs = []
for i in range(1, 13):
    path = Path(f"../results/nn/arctic-tests/all/monthly/{i}")
    run = next(path.iterdir())

    metrics_file = os.path.join(run, "metrics.json")
    with open(metrics_file) as f:
        runs.append(json.loads(f.read()))

metrics = {
    "mae": [],
    "rmse_cms": [],
    "skill": [],
    "acc": [],
    "mean_pred": [],
    "std": [],
}
for run in runs:
    for metric, value in run.items():
        if metric in metrics:
            metrics[metric].append(value)
            if metric == "rmse_cms":
                metrics[metrics] /= 100

fig, axs = plt.subplots(2, 3, figsize=(9, 6))
axs = axs.flatten()
fig.suptitle("Monthly Model Performance Metrics")

months = np.arange("1976-01", "1977-01", dtype="datetime64[M]")
titles = [
    "Mean Absolute Error",
    "RMSE",
    "Skill",
    "Anomaly Correlation Coefficient",
    "Mean Speed",
    "Std. Deviation",
]

for i, (metric, values) in enumerate(metrics.items()):
    metrics[metric] = np.array(values)

    axs[i].plot(months, metrics[metric])
    axs[i].set_title(titles[i])
    axs[i].tick_params(axis="x", rotation=45)
    axs[i].xaxis.set_major_locator(MonthLocator(bymonth=np.arange(1, 13)))
    axs[i].xaxis.set_major_formatter(DateFormatter("%b"))

fig.tight_layout()

eval_path = Path("../results/eval")
eval_path.mkdir(parents=True, exist_ok=True)
plt.savefig(eval_path / "figures.png", dpi=200)
