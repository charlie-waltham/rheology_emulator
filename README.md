# Rheology emulator

Here, we aim at deriving a cheap statistical emulator for the rheology solver in sea-ice models.

## Training an model
The best way to train a model is with the ```main.py``` script and a configuration file.

One example has is available out of the box (configuration in /configs/training/dv/dv-0.yaml).
```
cd code
python -m main --train --training_cfg dv/dv-0
```

That will create a results dir in /results/dv/dv-0/YYYYMMDD_HHMM/ with some quick look figures, the trained pytorch model, the dataloaders use, and a csv with validation data.
