import pandas as pd
from tensorflow.python.lib.io import file_io
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import logging
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-dir",  # handled automatically by AI Platform
    help="GCS location to write checkpoints and export models",
    required=False,
)
parser.add_argument(
    "--bucket_name",
    type=str,
    default="mw_ml_workshop",
    help="The Cloud Storage bucket to be used for process artifacts",
)

args = parser.parse_args()


# categorical columns contain data that need to be turned into numerical values before being used by XGBoost
CATEGORICAL_COLUMNS = (
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
)

# file_loc = "gs://" + str(args.bucket_name)
bucket_name=args.bucket_name


# load training set
# with file_io.FileIO(file_loc + "/census_income/data/adult.data", "r") as train_data:
with file_io.FileIO(f"gs://{bucket_name}/adult.data", "r") as train_data:
    raw_training_data = pd.read_csv(train_data)
# remove column we are trying to predict ('income') from features list
train_features = raw_training_data.drop("income", axis=1)
# create training labels list
train_labels = raw_training_data["income"] == " >50K"

# load test set
# with file_io.FileIO(file_loc + "/census_income/data/adult.test", "r") as test_data:
with file_io.FileIO(f"gs://{bucket_name}/adult.test", "r") as test_data:
    raw_testing_data = pd.read_csv(test_data, skiprows=[1])
# remove column we are trying to predict ('income') from features list
test_features = raw_testing_data.drop("income", axis=1)
# create training labels list
test_labels = raw_testing_data["income"] == " >50K."

# convert data in categorical columns to numerical values
encoders = {col: LabelEncoder() for col in CATEGORICAL_COLUMNS}
for col in CATEGORICAL_COLUMNS:
    train_features[col] = encoders[col].fit_transform(train_features[col])
for col in CATEGORICAL_COLUMNS:
    test_features[col] = encoders[col].fit_transform(test_features[col])

# load data into DMatrix object
dtrain = xgb.DMatrix(train_features, train_labels)
dtest = xgb.DMatrix(test_features)

logging.info("data loading complete")

# train XGBoost model
bst = xgb.train({"objective": "reg:logistic"}, dtrain, 20)
bst.save_model("./model.bst")

# upload the saved model file to Cloud Storage
subprocess.check_call(["gsutil", "cp", "model.bst", f"gs://{bucket_name}/model/"])
