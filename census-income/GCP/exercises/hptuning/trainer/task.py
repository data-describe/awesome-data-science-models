# import statements

import os

import xgboost as xgb
import hypertune

import argparse


def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
                        '--hp_parameter1',
                        required=True,
                        type=float,
                        help='dummy-example')
    parser.add_argument(
                        '--hp_parameter2',
                        required=True,
                        type=float,
                        help='dummy-example')
    parser.add_argument(
                        '--hp_parameter3',
                        required=True,
                        type=float,
                        help='dummy-example')

    args = parser.parse_args()
    return args


def build_model(
                hp_parameter1,
                hp_parameter2,
                hp_parameter3
               ):
    
    # read and prepare data
    
    # categorical columns contain data that need to be turned into numerical values before being used by XGBoost
    
    # choose classifier
    model = xgb.XGBClassifier(hp_parameter1=hp_parameter1,
                              hp_parameter2=hp_parameter2,
                              hp_parameter3=hp_parameter3)
    
    return model, train_features, train_labels, test_features, test_labels


def main():
    args = get_args()

    model, X_train, y_train, X_test, y_test = build_model(
                                                          args.hp_parameter1,
                                                          args.hp_parameter2,
                                                          args.hp_parameter3
                                                         )

    # model.fit etc.

    # Get predictions and scores

    # DEFINE METRIC
    hp_metric = <>  # metric to be optimized

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
                                            hyperparameter_metric_tag='roc_auc',  # e.g.
                                            metric_value=hp_metric,
                                           )

if __name__ == "__main__":
    main()