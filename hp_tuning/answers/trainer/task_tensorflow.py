import os
import sys
import json
import subprocess
import argparse as ap
import pandas as pd
import numpy as np
import datetime as dt

from . import model

# cli arguments
parser = ap.ArgumentParser()
parser.add_argument(
    '--shuffle_size',
    help='Shuffle buffer size for training data',
    default=10000,
    type=np.int64
)
parser.add_argument(
    '--batch_size',
    help='Batch size for training and validation data',
    default=32,
    type=np.int64
)
parser.add_argument(
    '--train_uri',
    help='URI to training data',
    default='gs://amazing-public-data/census_income/census_income_data_adult.data'
)
parser.add_argument(
    '--test_uri',
    help='URI to training data',
    default='gs://amazing-public-data/census_income/census_income_data_adult.test'
)
parser.add_argument(
    '--class_weight_ratio',
    help='Class weight ratio measured as majority class:minority class',
    default=1.0,
    type=np.float64
)
parser.add_argument(
    '--initial_lr',
    help='initial learning rate for training',
    default=1e-3,
    type=np.float64
)
parser.add_argument(
    '--lr_decay_param',
    help='controls how quickly the learning rate decays (higher=more slowly)',
    default=50.0,
    type=np.float64
)
parser.add_argument(
    '--hidden_depth',
    help='Number of hidden layers',
    default=2,
    type=np.int32
)
parser.add_argument(
    '--hidden_nodes',
    help='Number of nodes in each hidden layer',
    default=32,
    type=np.int32
)
parser.add_argument(
    '--hidden_activation',
    help='Activation for hidden layers',
    default='relu',
    type=str
),
parser.add_argument(
    '--kernel_regularizer',
    help='Kernel regularizer for hidden layers',
    default=None,
    type=str
)
parser.add_argument(
    '--optimizer',
    help='Model fit optimizer',
    default='Adam',
    type=str
)
parser.add_argument(
    '--epochs',
    help='Number of epochs to train on',
    default=3,
    type=np.int32
)
parser.add_argument(
    '--model_dir',
    help='Location where models are saved',
    default='local_models',
    type=str
)
parser.add_argument(
    '--model_prefix',
    help='Prefix for model name (name is appended with a timestamp)',
    default='model',
    type=str
)
parser.add_argument(
    '--metric',
    help='Metric used to measure model performance.',
    default='val_auc',
    type=str
)
parser.add_argument(
    '--run_type',
    help='Type of run (cloud or local)',
    default='local',
    choices=['local', 'cloud'],
    type=str
)
args, other_args = parser.parse_known_args()

# preprocess data
RANDOM_STATE = 5465
columns = (    
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
)
train = pd.read_csv(args.train_uri, names=columns, skiprows=1)
test = pd.read_csv(args.test_uri, names=columns, skiprows=2)
scaler = model.create_scaler(train)
train_prepped = model.preprocess(train, scaler)
test_prepped = model.preprocess(test, scaler)
test_prepped = test_prepped.reindex(columns=train_prepped.columns)
test_prepped = test_prepped.fillna(0)

train_tf = model.tf_dset_prep(train_prepped)
test_tf = model.tf_dset_prep(test_prepped, shuffle=False)

# train model
input_sz = len(train_prepped.columns) - 1

nn_model = model.build_nn_model(input_sz, 'model', args)
nn_model = model.fit_model(nn_model, train_tf, test_tf, args)

# if args.run_type == 'cloud':
#     # record trial performance (for hypertuning)
#     tf_config_str = os.environ.get('TF_CONFIG', 'No Config')
#     tf_config_dict = json.loads(tf_config_str)

#     # FOR TESTING: RECORD TF_CONFIG
#     # ==========================================
#     tf_config_fname = f'tf_confg_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
#     with open(tf_config_fname, 'a') as f:
#         json.dump(tf_config_dict, f)
#     res = subprocess.check_call(
#         ['gsutil', 'cp', tf_config_fname, os.path.join(args.model_dir, tf_config_fname)],
#         stderr=sys.stdout
#     )
#     # ==========================================

#     trial_n = tf_config_dict['task']['trial']

# # save model
#     model_name = f'{args.model_prefix}_trial{str(trial_n)}'
# else:
#     model_name = f'{args.model_prefix}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
# model.save_model(nn_model, os.path.join(args.model_dir, model_name))