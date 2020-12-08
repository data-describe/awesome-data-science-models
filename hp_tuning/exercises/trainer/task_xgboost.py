import os
import sys
import json
import subprocess
import hypertune
import argparse as ap
import pandas as pd
import numpy as np
import datetime as dt
import xgboost as xgb

from sklearn.metrics import roc_auc_score
from . import model

# cli args
parser = ap.ArgumentParser()
# parser.add_argument(
#     '--batch_size',
#     help='Batch size for training and validation data',
#     default=32,
#     type=np.int64
# )
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
    '--num_boost_round',
    help='Number of boosters to train',
    default=10,
    type=np.int32
)
parser.add_argument(
    '--learning_rate',
    help='Boosting learing rate',
    default=0.3,
    type=np.float64
)
parser.add_argument(
    '--max_depth',
    help='Max tree depth',
    default=6,
    type=np.int32
)
parser.add_argument(
    '--class_weight_ratio',
    help='Class weight ratio measured as majority class:minority class',
    default=1.0,
    type=np.float64
)
parser.add_argument(
    '--metric',
    help='Metric used to measure model performance.',
    default='auc',
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

X_train = train_prepped.loc[:, train_prepped.columns.difference(['income_50k'])].values
y_train = train_prepped.income_50k.values.tolist()
X_test = test_prepped.loc[:, test_prepped.columns.difference(['income_50k'])].values
y_test = test_prepped.income_50k.values.tolist()

train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

# train model
other_params = {
    "objective": 'binary:logistic',
    "max_depth": args.max_depth,
    "learning_rate": args.learning_rate,
    "scale_pos_weight": args.class_weight_ratio,
    "eval_metric": args.metric,
    # "verbosity": 0,
    "seed": RANDOM_STATE  # for consistency
}

xgb_model = xgb.train(
    other_params,
    train_data, 
    evals=[(train_data, 'train'), (test_data, 'test')],
    num_boost_round=args.num_boost_round,
    early_stopping_rounds=8
)

# evaluate model
y_preds = xgb_model.predict(test_data)
auc_score = roc_auc_score(y_test, y_preds)

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='htune_ROC_AUC1',
    metric_value=auc_score,
)

if args.run_type == 'cloud':
    # record trial performance (for hypertuning)
    tf_config_str = os.environ.get('TF_CONFIG', 'No Config')
    tf_config_dict = json.loads(tf_config_str)
    trial_n = tf_config_dict['task']['trial']
    # save model
    model_name = f'{args.model_prefix}_trial{str(trial_n)}.bst'
    xgb_model.save_model(model_name)  # locally
    # save to cloud location
#     model_path = os.path.join(args.model_dir, model_name)
#     res = subprocess.check_call(['gsutil', 'cp', model_name, model_path], stderr=sys.stdout)
# else:    
#     # save model
#     model_name = f'{args.model_prefix}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.bst'
#     model_path = os.path.join(args.model_dir, model_name)
#     xgb_model.save_model(model_path)