# Rheology Emulator

Here, we aim at deriving a cheap statistical emulator for the rheology solver in sea-ice models.

## Installation
Ensure CUDA 12.6 is installed on your system.

Install with [uv](https://docs.astral.sh/uv/):
```
uv venv
uv sync
source .venv/bin/activate
```
Alternatively, install with pip:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets
An Arctic dataset should be provided at `data/processed/arctic/all/all.zarr`. Likewise for the Antarctic at `data/processed/anatarctic/all/all.zarr`.

## Training
An example of training an Arctic model with config located at `configs/training/arctic-tests/all/all.yaml` is shown below.

```
cd code
python main.py --train --training_cfg arctic-tests/all/all
```

After training this will create a folder in `results/nn`.

## Testing and Evaluation
Testing will run a trained model on its test set, from which results can then be evaluated.

```
cd code
python main.py --test --eval_path ../results/nn/...
python main.py --evaluate --eval_path ../results/nn/...
```

This will create single-step evaluation plots in the results folder.

## Other experiments
`autoregression.py` uses a trained model to autoregress on a dataset. It is recommended to use a continuous dataset with no time gaps for this to work properly.
```
cd code
python autoregression.py <model> <dataset>
```
This will create an evaluation plot in `/results/autoregression`.

`monthly.py` is a utility to train all model configs in a given config folder. It will skip training any models which already have a results folder in its directory, which allows for easy pausing/resuming.
```
cd code
python monthly.py <tests_path> <results_dir>
```
