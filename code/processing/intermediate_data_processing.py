import sys
import xarray as xr
import numpy as np

def process_intermediate_data(input_file, output_file, label, threshold):

    chunk_size = 10000

    print('Loading ' + input_file)
    raw = xr.open_zarr(input_file)
    raw = raw.chunk({'z': chunk_size})  
    for var in raw.data_vars:
        raw[var] = raw[var].chunk({'z': chunk_size})

    dlab = raw['labels'].sel(label=label) - raw['features'].sel(feature=label)

    lab_mask = abs(dlab) > threshold
    
    print('Masking %s to %s' % (label, threshold))
    # processed = raw.where(lab_mask, drop=False)

    mask = lab_mask.compute()
    indicies = np.where(mask)[0]
    processed = raw.isel(z=indicies)

    print(processed)

    print('Saving to ' + output_file)
    # Rechunk data variables
    processed = processed.chunk({'z': 10000})

    # Rechunk coordinates explicitly if they are large arrays
    for coord in processed.coords:
        if 'z' in processed[coord].dims:
            processed[coord] = processed[coord].chunk({'z': 10000})

    # Clear encoding for all variables and coordinates
    for var in processed.data_vars:
        processed[var].encoding.pop('chunks', None)
    for coord in processed.coords:
        processed[coord].encoding.pop('chunks', None)
        
    processed.to_zarr(output_file, mode='w')

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: python intermediate_data_processing.py <input_file> <output_file> <label> <threshold>')
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    label = sys.argv[3]
    threshold = float(sys.argv[4])

    process_intermediate_data(input_file, output_file, label, threshold)