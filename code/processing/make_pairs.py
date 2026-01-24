'''
This script makes a zarr with pairs of features at time t and labels at time t+1 from SI3 output files.
The desired features and labels, and the list of model output files, can be user specified via a config file.

The config should be stored in configs/data_gathering and should be yml formatted.
An example working config is evp_120itr_12day.yml

Usage: python make_pairs.py <config_name>
Example: python make_pairs.py evp_120itr_12day

Alternatively, if more memory is needed, use the provided make_pairs.sh script to submit via srun with 128G memory:
./make_pairs.sh <config_name>
'''
import yaml
import sys
import shutil 

import xarray as xr
import numpy as np

from pathlib import Path

def read_cfg(cfg_name):
    ''' Read config to dict, with some logic to deal with +/- the .yaml extension. '''
    if cfg_name.endswith('.yaml'):
        cfg_name = cfg_name[:-5]
    cfg_path = '../configs/data_gathering/' + cfg_name + '.yaml'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def make_filelist(cfg):
    ''' Make list of input files from config. '''
    raw_data_path = Path(cfg['raw_data_path'], cfg['data_label'])
    filelist = [raw_data_path / fname for fname in cfg['files']]
    return filelist

def make_pairs(filelist, features, labels, stack_pairs=False):
    ''' 
    Make pairs of features at time t and labels at time t+1 from list of input files, features, and labels.

    Returns an xarray Dataset with dimensions:
    - pair: pairs of features and labels 
    - feature: features at time t (e.g. siconv, sivelv)
    - label: labels at time t+1 (e.g. sivelv)
    - y, x: spatial dimensions
    
    The timestamp, lat, and lon coordinates are also included.
    '''
    # TODO: If needed, make chunking behavior specifiable
    pair_ds_list = []

    for input_path in filelist:
        ds = xr.open_dataset(input_path, chunks={'time_counter': 1})
        n_pairs = len(ds.time_counter) - 1

        features_arr = xr.concat(
            [ds[var].isel(time_counter=slice(0, -1)) for var in features], dim="feature"
        ).transpose("time_counter", "feature", "y", "x").data  # Dask array

        labels_arr = xr.concat(
            [ds[var].isel(time_counter=slice(1, None)) for var in labels], dim="label"
        ).transpose("time_counter", "label", "y", "x").data  # Dask array

        pair_ds = xr.Dataset(
            data_vars={
                "features": (["pair", "feature", "y", "x"], features_arr),
                "labels": (["pair", "label", "y", "x"], labels_arr),
            },
            coords={
                "pair": np.arange(n_pairs),
                "feature": features,
                "label": labels,
                "y": ds.y.values,
                "x": ds.x.values,
                "lat": (["y", "x"], ds.nav_lat.values),
                "lon": (["y", "x"], ds.nav_lon.values),
                "time_features": (["pair"], ds.time_counter.values[:-1]),
                "time_labels": (["pair"], ds.time_counter.values[1:]),
            }
        )
        pair_ds_list.append(pair_ds)

    # Concatenate along the 'pair' dimension if more than one file
    if len(pair_ds_list) > 1:
        paired_ds = xr.concat(pair_ds_list, dim="pair")
        # Reset the 'pair' coordinate to be a continuous range from 0 to total_pairs-1
        total_pairs = paired_ds.sizes['pair']
        paired_ds = paired_ds.assign_coords(pair=np.arange(total_pairs))
    else:
        paired_ds = pair_ds_list[0]

    if stack_pairs:
        # Stack the pair dimension into the spatial dimensions to create a larger dataset
        paired_ds = paired_ds.stack(z=('pair', 'y', 'x'))
        paired_ds = paired_ds.drop_vars('pair')
        paired_ds = paired_ds.rename({'z': 'pair'})

    return paired_ds

def save_pairs(paired_ds, output_path, data_label):
    ''' Save paired dataset to zarr format in location specified by config. '''
    save_path = Path(output_path) / f'pairs_{data_label}.zarr'
    paired_ds.to_zarr(save_path, mode='w')
    return save_path

def main(cfg):
    ''' Read config, make pairs, and save to zarr. '''

    filelist = make_filelist(cfg)
    features = cfg['features']
    labels = cfg['labels']
    output_path = cfg['output_path']
    data_label = cfg['data_label']

    print('Making pairs for: %s' % data_label)
    print('Features: %s' % features)
    print('Labels: %s' % labels)

    paired_ds = make_pairs(filelist, features, labels)
    print('Made pairs for: %s' % data_label)
    print(paired_ds)

    saved_to = save_pairs(paired_ds, output_path, data_label)
    print('Saved pairs to: %s' % saved_to)

    # TODO: move log file to saved_to
    logfile = f'{data_label}_make_pairs.log' # needs to match make_pairs.sh
    logfile_path = Path(output_path, logfile)
    print('Log file: %s' % logfile_path)
    shutil.move(Path('logs', logfile), logfile_path)

if __name__ == "__main__":
    # Example usage: python make_pairs.py evp_120itr_12day
    cfg_name = sys.argv[1]
    # Parse config to dictionary
    cfg = read_cfg(cfg_name)
    # Make and save the pairs of features and labels
    main(cfg)