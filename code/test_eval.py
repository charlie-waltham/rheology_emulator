import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np

runs = []
for i in range(1, 13):
    path = Path(f"../results/nn/charlie/arctic-test-suite/all/{i}")
    run = next(path.iterdir())
    
    metrics_file = os.path.join(run, "metrics.json")
    with open(metrics_file, "r") as f:
        runs.append(json.loads(f.read()))

metrics = {}
for run in runs:
    for metric, value in run.items():
        if metric not in metrics:
            metrics[metric] = []
        metrics[metric].append(value)

fig, axs = plt.subplots(1, len(metrics), sharex=True, figsize=(20, 5))
fig.tight_layout()
months = np.arange("1976-01", "1977-01", dtype="datetime64[M]")

for i, (metric, values) in enumerate(metrics.items()):
    metrics[metric] = np.array(values)

    axs[i].plot(months, metrics[metric])
    axs[i].set_title(metric)
    axs[i].xaxis.set_major_locator(MonthLocator(bymonth=np.arange(1, 13)))
    axs[i].xaxis.set_major_formatter(DateFormatter('%b'))

eval_path = Path("../results/eval")
eval_path.mkdir(parents=True, exist_ok=True)
plt.savefig(eval_path / "figures.png")
