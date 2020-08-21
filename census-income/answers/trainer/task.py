import pandas as pd
from tensorflow.python.lib.io import file_io
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import logging
import subprocess
import argparse


# these are the column labels from the census data files
COLUMNS = (
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income-level'
)

# categorical columns contain data that need to be turned into numerical values before being used by XGBoost
CATEGORICAL_COLUMNS = (
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country'
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--job-dir',  # handled automatically by AI Platform
    help='GCS location to write checkpoints and export models',
    required=False
)
parser.add_argument(
    '--bucket_name',
    type=str,
    default='mw_ml_workshop',
    help='The Cloud Storage bucket to be used for process artifacts'
)

args = parser.parse_args()

file_loc = 'gs://'+str(args.bucket_name)


# load training set
with file_io.FileIO(file_loc+ '/census_income/data/adult.data', 'r') as train_data:
    raw_training_data = pd.read_csv(train_data, header=None, names=COLUMNS)
# remove column we are trying to predict ('income-level') from features list
train_features = raw_training_data.drop('income-level', axis=1)
# create training labels list
train_labels = (raw_training_data['income-level'] == ' >50K')



# load test set
with file_io.FileIO(file_loc+'/census_income/data/adult.test', 'r') as test_data:
    raw_testing_data = pd.read_csv(test_data, names=COLUMNS, skiprows=1)
# remove column we are trying to predict ('income-level') from features list
test_features = raw_testing_data.drop('income-level', axis=1)
# create training labels list
test_labels = (raw_testing_data['income-level'] == ' >50K.')

# convert data in categorical columns to numerical values
encoders = {col:LabelEncoder() for col in CATEGORICAL_COLUMNS}
for col in CATEGORICAL_COLUMNS:
    train_features[col] = encoders[col].fit_transform(train_features[col])
for col in CATEGORICAL_COLUMNS:
    test_features[col] = encoders[col].fit_transform(test_features[col])

# load data into DMatrix object
dtrain = xgb.DMatrix(train_features, train_labels)
dtest = xgb.DMatrix(test_features)

logging.info('data loading complete')

# train XGBoost model
bst = xgb.train({'objective':'reg:logistic'}, dtrain, 20)
bst.save_model('./model.bst')

# upload the saved model file to Cloud Storage
subprocess.check_call(['gsutil', 'cp', 'model.bst', file_loc+'/census_income/model/'])





