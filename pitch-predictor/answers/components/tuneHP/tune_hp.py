# libraries
import argparse
from google.cloud import storage
import json
import logging
import numpy as np
import pandas as pd
import os

from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin

import xgboost as xgb

from sklearn.metrics import roc_auc_score



def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pitch_type', dest='pitch_type', default='SI', help='Select the pitch type to evaluate')
    '''
    parser.add_argument('--project', dest='project', default='{{ GCP_PROJECT }}', help='Select the gcp project to run this job')
    parser.add_argument('--staging_location', dest='staging_location', default='gs://dataflow-holding/dataflow_stage/', help='Select the staging location for this job')
    parser.add_argument('--temp_location', dest='temp_location', default='gs://dataflow-holding/dataflow_tmp/', help='Select the temp location for this job')
    parser.add_argument('--setup_file', dest='setup_file', default='/root/setup.py', help='Config options for the pipeline')
    '''

    known_args, _ = parser.parse_known_args(argv)

    # define the pitch type 
    pitch_type = known_args.pitch_type


    def objective(params):
        '''Objective function for Gradient Boosting Machine Hyperparameter Tuning'''

        # determine AUC with current parameters
        bst = xgb.train(params=params, dtrain=dtrain)
        preds = bst.predict(dval)
        auc = roc_auc_score(val_labels, preds)

        # Loss must be minimized
        loss = 1 - auc

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}


    # download the  data
    storage_client = storage.Client()
    bucket_name = '{{ GCP_PROJECT }}-pitch-data'

        # train
    source_blob_name = pitch_type + '/train.csv'
    destination_file_name = 'train.csv'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    df_train = pd.read_csv('train.csv')

        # val
    source_blob_name = pitch_type + '/val.csv'
    destination_file_name = 'val.csv'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    df_val = pd.read_csv('val.csv')


    # prepare data for the xgboost model
        # training
    train_labels = df_train[pitch_type] == 1
    train_features = df_train.drop(pitch_type, axis=1)
    dtrain = xgb.DMatrix(train_features, train_labels)

        # validation
    val_labels = df_val[pitch_type] == 1
    val_features = df_val.drop(pitch_type, axis=1)
    dval = xgb.DMatrix(val_features, val_labels)

    # Define the search space
    space = {
        'map': hp.uniform('map', 0, 1), # uniform 
        'max_depth': hp.choice('max_depth', np.arange(1, 100+1, dtype=int)),  # discrete uniform
        'min_child_weight': hp.uniform('min_child_weight', 0, 100),
        'lambda': hp.uniform('lambda', 0, 100),
        'alpha': hp.uniform('alpha', 0, 100),
        'num_boost_round': hp.choice('num_boost_round', np.arange(100,1000 , dtype=int))  # discrete uniform
    }


    # begin hyperparameter tuning
    bayes_trials = Trials()
    MAX_EVALS = 10

    # optimize
    best_params = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)

    # push params to cloud storage
    best_params_dump = json.dumps(best_params)

    destination_blob_name = pitch_type + '/params.json'


    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(best_params_dump)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    run()



