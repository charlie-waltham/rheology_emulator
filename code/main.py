import argparse 
import logging
import numpy
import random
import shutil
import torch
import os
import pprint
import yaml

from datetime import datetime

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help="Enable training mode")
    parser.add_argument('--evaluate', action='store_true', help="Enable evaluation mode")
    parser.add_argument('--csv_path', type=str, default=None, help="Path to CSV with 'True Values' and 'Predictions' (defaults to results_path/ytrue_ypred_val.csv)")

    parser.add_argument('--save_data', type=bool, default=False, help="Whether to save the training, validation, and test datasets after splitting")
    parser.add_argument('--save_val', type=bool, default=False, help="Whether to save the validation ytrue, ypred, and inputs after training")

    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")

    # Run configuration file
    parser.add_argument('--training_cfg', type=str, help="Name of the training configuration file to load")
    parser.add_argument('--parameters', type=str, help="Name of the parameters file to load")

    # Optional arguments for training
    parser.add_argument('--zarr_fmt', type=str, default='fmt1', help="fmt1: zarr with separate pairs (via make_pairs.py), fmt2: zarr with concatonated pairs (via make_pairs_2.py)")
    parser.add_argument('--difference_labels', type=bool, default=False, help="Whether to use differenced (l(t+1) - l(t) or absolute labels (l(t+1))")
    parser.add_argument('--pairs_path', type=str, required=False, help="Path to the pairs data file for training - should be a zarr")
    parser.add_argument('--results_path', type=str, required=False, help="Path to where model results will be saved")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--val_fraction', type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument('--test_fraction', type=float, default=0.1, help="Fraction of data to use for testing")
    parser.add_argument('--scale_features', action='store_true', default=True, help="Enable feature and label scaling for training")
    parser.add_argument('--architecture', type=int, default=0, help="Full path to yaml with architecture")
    parser.add_argument('--shorten_dataset', type=int, default=None, help="Whether to shorten the dataset to 10,000 samples for quick testing/debugging")
    parser.add_argument('--sequential', action='store_true', default=False, help="Whether to use sequential data loading (for RNNs)")
    parser.add_argument('--hemisphere', type=str, default="north", help="Hemisphere to plot MAE maps for")

    return vars(parser.parse_args())

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file="main.log"):

    # Delete the log file if it exists already
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # Optional: Also log to console

def load_training_config(confg_name, arguments):
    
    confg_path = '../configs/training/' + confg_name + '.yaml'
    with open(confg_path, 'r') as file:
        config = yaml.safe_load(file)
    
    arguments.update(config)
    arguments['training_config_path'] = confg_path

def setup_results(args):
    # Get the current time
    current_time = datetime.now()

    # Format the time as yyyymmdd_HHMM
    current_time = current_time.strftime("%Y%m%d_%H%M")

    args['results_path'] = args['results_path'] + current_time +'/'

    os.makedirs(args['results_path'], exist_ok=True) 
    logging.info(f"Results directory set up at {args['results_path']}")

    if args['save_data']:
        os.makedirs(args['results_path'] + 'data_splits/', exist_ok=True) 
        logging.info(f"Data splits directory set up at {args['results_path']}data_splits/")


def train_model(args):
    # Setup logging for training
    setup_logging(log_file='train_model.log')  # Set up logging

    # Load in config file for training, overwriting any duplications in args
    if not args['training_cfg'] is None:
        load_training_config(args['training_cfg'], args)

    logging.info(pprint.pformat(args))

    setup_results(args)  # Set up results directory

    if not args['train']:
        print("Training mode is not enabled. Use --train to enable it.")
        return

    if not args['pairs_path'] or not args['results_path']:
        print("Both --pairs_path and --results_path must be provided for training.")
        return

    logging.info("Training model...")
    from src.train_nn import train_save_eval
    train_save_eval(args)
    logging.shutdown()

    # Move log file to results directory
    shutil.move('train_model.log', args['results_path']+'train_model.log')
    shutil.copy(args['training_config_path'], args['results_path']+'used_training_config.yaml')

def main():
    args = parse_arguments()

    # Set the random seed for reproducibility
    set_seed(args['seed'])

    print(args)

    # Check if multiple modes are enabled
    if args['train'] + args['evaluate'] > 1:
        print("Error: Only one mode can be enabled at a time. Please choose one of --train or --evaluate.")
        return

    elif args['train']:
        train_model(args)

    # TODO: write eval code to work on test set
    elif args['evaluate']:
        if not args['results_path']:
            print("Error: --results_path must be provided for evaluation.")
            return

        csv_path = args['csv_path'] if args['csv_path'] else os.path.join(args['results_path'], 'ytrue_ypred_val.csv')
        if not os.path.exists(csv_path):
            print(f"Error: CSV not found at {csv_path}")
            return

        from src.evaluate import evaluate_and_save

        saved = evaluate_and_save(csv_path, args['results_path'])
        print("Saved figures:")
        for name, path in saved.items():
            print(f"  {name}: {path}")
        

if __name__ == "__main__":
    main()
        