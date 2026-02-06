# TODO: Make definition of features include t and t+1 variables for everything except the target
# so like siconc(t), siconc(t+1), sivelv(t) -> sivelv(t+1)
# 
import argparse as ap
import random
import torch
import yaml

import pandas as pd
import xarray as xr
import torch.nn as nn
import torch.optim as optim

from . import models

# Parse command line arguments
def parse_args(dummy_args=False):
    '''Parse command line arguments.
    
    Args: 
        dummy_args: bool, if True, return dummy arguments for testing
        
    Returns:
        args: dict, dictionary of arguments
    '''
    if dummy_args:
        args = {'model_type': 'SGDRegressor',
            'test_fraction': 0.1,
            'suite': 'u-cn464',
            'version': 'raw_v0',
            'features': ['siconc', 'sithic', 'utau_ai', 'utau_oi', 'vtau_ai', 'vtau_oi'],#,'sishea', 'sistre'],
            'labels': ['sig1_pnorm'],
            'flatten': True,
            'StandardScalar': True
               }
        return args
    else:
        parser = ap.ArgumentParser(description='Train a regression model')
        parser.add_argument('--model_type', type=str, default='SGDRegressor', help='Type of model to use')
        parser.add_argument('--test_fraction', type=float, default=0.1, help='Fraction of data to use for testing')
        parser.add_argument('--suite', type=str, default='u-cn464', help='Name of suite to use')
        parser.add_argument('--version', type=str, default='raw_v0', help='Version of suite to use')
        parser.add_argument('--features', type=str, default='siconc,sithic,utau_ai,utau_oi,vtau_ai,vtau_oi,sishea,sistre,sig1_pnorm,sig2_pnorm,sivelv,sivelu', help='List of features to use')
        parser.add_argument('--labels', type=str, default='sig1_pnorm', help='List of labels to use')
        parser.add_argument('--flatten', type=bool, default=True, help='Flatten data')
        parser.add_argument('--StandardScalar', type=bool, default=True, help='Standardise data')
        parser.add_argument('--tune', type=bool, default=True, help='Tune hyperparameters')
        parser.add_argument('--data_points', type=int, default=1000, help='Data points to feed model (less for quick testing)')
        parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
        parser.add_argument('--validation_fraction', type=float, default=0.2, help='Fraction of train data to validate on')

        args = parser.parse_args()
        args.features = args.features.split(',')     
        args.labels = args.labels.split(',')         
                                                                                                                                                                                                                                                     
        return vars(args)

# Load data function
def load_data(args):
    '''
    Load raw data and make feature (X(t)) - label (y(t+1)) pairs.
    '''
    suite = args['suite']
    version = args['version']
    filename = '../data/' + suite + '/raw/' + suite + '_' + version + '.nc'
    pairs = make_feature_label_pairs(filename, args['features'], args['labels'], flatten=args['flatten'])
    return pairs

def make_feature_label_pairs(filename, feature_names, label_names, flatten=False, feature_names_tp1=None):
    data = xr.open_dataset(filename)
    if flatten:
        data = data.stack(xy=('x', 'y'))
        data = data.dropna(dim='xy')
    features = data[feature_names]
    # if feature_names_tp1 is not None:
    #     features_tp1 = data[feature_names_tp1]
    labels = data[label_names]
    pairs = []
    for itime in range(features.time_counter.size - 1):
        feature = features.isel(time_counter=itime)
        # if feature_names_tp1 is not None:
        #     feature_tp1 = features_tp1.isel(time_counter=itime+1)
        #     # add feature_tp1 to feature as extra variables
        #     feature = xr.concat([feature, feature_tp1], dim='time_counter')
        label = labels.isel(time_counter=itime+1)
        pairs.append((feature, label))
    return pairs

# Make test and train sets
def make_test_train(pairs, args):
    '''
    Split pairs into training and testing sets, and concatonate the pairs into one dataframe.

    Args:
        pairs: list, list of pairs of feature and label dataframes
        args: dict, dictionary of arguments

    Returns:
        train_X: dataframe, training feature data
        train_y: dataframe, training label data
        test_X: dataframe, testing feature data
        test_y: dataframe, testing label data
    '''
    pairs_df = []
    for pair in pairs:
        try:
            pairs_df.append( 
                            (pd.DataFrame(pair[0].to_array().values.T), 
                            pd.DataFrame(pair[1].to_array().values.reshape(-1, ))) 
                            )
        except Exception:
            pairs_df.append( 
                            (pd.DataFrame(pair[0].to_array().values.T), 
                            pd.DataFrame(pair[1].values.reshape(-1, )))
            )

    # randomly select 10% of the pairs as a test holdout
    n = len(pairs_df)
    n_test = int(args['test_fraction']*n)
    test_indices = random.sample(range(n), n_test)
    train_indices = [i for i in range(n) if i not in test_indices]

    train_pairs = [pairs_df[i] for i in train_indices]
    test_pairs = [pairs_df[i] for i in test_indices]

    train_X = pd.concat([pair[0] for pair in train_pairs], axis=0)
    train_y = pd.concat([pair[1] for pair in train_pairs], axis=0)
    test_X = pd.concat([pair[0] for pair in test_pairs], axis=0)
    test_y = pd.concat([pair[1] for pair in test_pairs], axis=0)
    
    print('Train shape initial: ', train_X.shape, train_y.shape)
    print('Test shape initial: ', test_X.shape, test_y.shape)

    return train_X, train_y, test_X, test_y


def reduce_data_points(train_X, train_y, n_data_points):
    '''
    Reduce the number of data points in the training set for ease of model development. TODO: do this better with batch training

    Args:
        train_X: dataframe, training feature data
        train_y: dataframe, training label data
        n_data_points: int, number of data points to keep

    Returns:
        train_X: dataframe, training feature data reduced
        train_y: dataframe, training label data reduced
    '''
    if n_data_points > 0:
        indicies = random.sample(range(len(train_X)), n_data_points)
        train_X = train_X.iloc[indicies]
        train_y = train_y.iloc[indicies]
    return train_X, train_y

def make_validation(train_X, train_y, validation_fraction):
    '''
    Make validation set from training set.

    Args:
        train_X: dataframe, training feature data
        train_y: dataframe, training label data
        validation_fraction: float, fraction of data to use for validation

    Returns:
        train_X: dataframe, training feature data less validation data
        train_y: dataframe, training label data less validation data
        val_X: dataframe, validation feature data
        val_y: dataframe, validation label data
    '''
    n = len(train_X)
    n_val = int(validation_fraction*n)
    val_indices = random.sample(range(n), n_val)
    train_indices = [i for i in range(n) if i not in val_indices]

    val_X = train_X.iloc[val_indices]
    val_y = train_y.iloc[val_indices]
    train_X = train_X.iloc[train_indices]
    train_y = train_y.iloc[train_indices]

    print('Train shape reduced: ', train_X.shape, train_y.shape)
    print('Validation shape: ', val_X.shape, val_y.shape)

    return train_X, train_y, val_X, val_y

def define_nn(architecture, n_features, n_labels, device):
    layer_list = nn_layer_list(architecture)
    layer_list = match_io_dims(layer_list, n_features, n_labels)
    model = build_model_from_layers(layer_list)
    # Parallelize model if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model

def nn_layer_list(config_path='../configs/nn_architecture/base.yaml'):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)['model']
    layer_list = []
    for item in config:
        layer_list.append([config[item]['type'], config[item]['args']])
    return layer_list

def match_io_dims(layer_list, n_features, n_labels):
    layer_list[0][1][0] = n_features  # Set input dimension of the first layer
    layer_list[-1][1][1] = n_labels  # Set output dimension of the last layer
    return layer_list

def build_model_from_layers(layer_list):
    # Create a list to hold the layers
    layers = []

    # Iterate through the layer definitions
    for layer_type, args in layer_list:
        # Get the layer class from nn
        layer_class = getattr(torch.nn, layer_type, None)
        # If not found, then look in models.py
        if layer_class is None:
            layer_class = getattr(models, layer_type)
        # Instantiate the layer with the provided arguments
        layers.append(layer_class(*args))

    # Create the neural network using nn.Sequential
    model = torch.nn.Sequential(*layers)

    return model

def nn_options(model, parameters='../configs/parameters/nn_base.yaml'):
    """
    Define the loss function, optimizer, and number of epochs based on a YAML configuration.
    """
    if parameters is None:
        raise ValueError("An architecture YAML file must be provided.")

    # Load the YAML configuration
    with open(parameters, "r") as f:
        config = yaml.safe_load(f)

    # Define the loss function
    loss_type = config.get("loss", "MSELoss")  # Default to MSELoss if not specified
    if loss_type == "ShrinkageLoss":
        from .loss import ShrinkageLoss
        criterion = ShrinkageLoss(
            a=config.get("shrinkage_a", 10),
            c=config.get("shrinkage_c", 0.2)
        )
    else:
        criterion = getattr(nn, loss_type)()

    # Define the optimizer
    optimizer_type = config.get("optimizer", "Adam")  # Default to Adam if not specified
    lr = config.get("learning_rate", 0.001)  # Default learning rate
    optimizer_class = getattr(optim, optimizer_type)
    optimizer = optimizer_class(model.parameters(), lr=lr)

    # Define the number of epochs
    n_epochs = config.get("epochs", 10)  # Default to 10 epochs

    return criterion, optimizer, n_epochs