# import statements

import os
import json

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import hypertune

import subprocess
import argparse


def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
                        '--learning_rate',
                        required=True,
                        type=float,
                        help='learning rate')
    parser.add_argument(
                        '--max_depth',
                        required=True,
                        type=int,
                        help='maximum depth of boosted tree')
    parser.add_argument(
                        '--scale_pos_weight',
                        required=True,
                        type=int,
                        help='weights ratio of negative to positive classes')

    args = parser.parse_args()
    return args


def build_model(
                learning_rate,
                max_depth,
                scale_pos_weight
               ):
    
    train_data = "gs://aaa-workshop-anurag/census_income/adult.data"
    raw_training_data = pd.read_csv(train_data)
    # remove column we are trying to predict ('income') from features list
    train_features = raw_training_data.drop("income",
                                            axis=1)
    # create training labels list
    train_labels = raw_training_data["income"] == " >50K"
    
    test_data = "gs://aaa-workshop-anurag/census_income/adult.test"
    raw_testing_data = pd.read_csv(test_data, skiprows=[1])
    # remove column we are trying to predict ('income') from features list
    test_features = raw_testing_data.drop("income",
                                          axis=1)
    # create labels list
    test_labels = raw_testing_data["income"] == " >50K."
    
    # categorical columns contain data that need to be turned into numerical values before being used by XGBoost
    CATEGORICAL_COLUMNS = (
                           "workclass",
                           "education",
                           "marital-status",
                           "occupation",
                           "relationship",
                           "race",
                           "sex",
                           "native-country"
                           )
    
    # convert data in categorical columns to numerical values
    encoders = {col: LabelEncoder() for col in CATEGORICAL_COLUMNS}
    
    for col in CATEGORICAL_COLUMNS:
        train_features[col] = encoders[col].fit_transform(train_features[col])
    
    for col in CATEGORICAL_COLUMNS:
        test_features[col] = encoders[col].fit_transform(test_features[col])
        
    model = xgb.XGBClassifier(learning_rate=learning_rate,
                              max_depth=max_depth,
                              scale_pos_weight=scale_pos_weight)
    
    return model, train_features, train_labels, test_features, test_labels


def main():
    args = get_args()

    model, X_train, y_train, X_test, y_test = build_model(
                                                          args.learning_rate,
                                                          args.max_depth,
                                                          args.scale_pos_weight
                                                         )

    model.fit(X_train, y_train)

    predictions = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test,
                            predictions)

    # DEFINE METRIC
    hp_metric = roc_auc  # metric to be optimized

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
                                            hyperparameter_metric_tag='roc_auc',
                                            metric_value=hp_metric,
                                           )

if __name__ == "__main__":
    main()
    



