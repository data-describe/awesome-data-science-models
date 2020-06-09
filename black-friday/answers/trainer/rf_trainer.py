import argparse
import datetime

from google.cloud import storage
import joblib
import json
import logging
import numpy as np
import os
import pandas as pd
import pandas_gbq
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import subprocess
import sys

import trainer.create_data_func as create_data_func
import trainer.hp_tuning as hp_tuning


def train_and_evaluate(args):
    # confirm whether data needs to be created for training
    if args.create_data == True:
        create_data_func.create_data_func(args.project_id, args.bucket_name, args.dataset_id)

    # download the data
    storage_client = storage.Client(project=args.project_id) 
    bucket = storage_client.get_bucket(args.bucket_name) 

        # x train
    blob = bucket.blob(args.train_file_x)
    blob.download_to_filename(args.train_file_x)
    x_train_df = pd.read_csv(args.train_file_x,header=None)

        # x test
    blob = bucket.blob(args.test_file_x)
    blob.download_to_filename(args.test_file_x)
    x_test_df = pd.read_csv(args.test_file_x,header=None)

        # y train
    blob = bucket.blob(args.train_file_y)
    blob.download_to_filename(args.train_file_y)
    y_train_df = pd.read_csv(args.train_file_y,header=None)

        # y test
    blob = bucket.blob(args.test_file_y)
    blob.download_to_filename(args.test_file_y)
    y_test_df = pd.read_csv(args.test_file_y,header=None)

    if args.hp_tune == True:
        logging.info('Started hyperparameter tuning')

        best_params = hp_tuning.complete_hp_tuning(x_train_part=x_train_df, y_train_part=y_train_df, project_id=args.project_id, bucket_name=args.bucket_name, num_iterations=args.num_hp_iterations)
        logging.info('Completed hyperparameter tuning with params {}'.format(str(best_params)))
    else:
        # download turning parameters from cloud storage
        blob = bucket.blob('best_params.json')
        blob.download_to_filename('best_params.json')

        with open('best_params.json') as json_file:
            best_params = json.load(json_file)

        logging.info('Parameters loaded {}'.format(str(best_params)))

    # fit a model on the entire training set with the best parameters
    logging.info('Fitting model across whole training set')
    rf_model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'], min_samples_leaf=best_params['min_samples_leaf'],  max_features=best_params['max_features'])
    rf_model.fit(x_train_df, y_train_df)

    # export the classifier to a file
    logging.info('Exporting model to Cloud Storage')
    model_filename = 'model.pkl'
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(rf_model, model_file)

    # upload the saved model file to Cloud Storage
    gcs_model_path = os.path.join('gs://', args.bucket_name, 'black_friday_{}'.format(args.job_id), model_filename)
    subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path])

    # assess the model accuracy
        # make predictions on the test set
    logging.info('Assessing model accuracy')
    y_pred = rf_model.predict(x_test_df)
    total_preds = 0
    total_correct = 0
    for i in range(0, y_pred.shape[0]):
        total_preds += 1

        if np.array_equal(y_pred[i], y_test_df.values[i]):
            total_correct += 1

    accuracy = str(round((total_correct / total_preds) * 100))
    logging.info('Predictions correct for {}% of test samples'.format(accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-file-x',
        type=str,
        help='Training file local or GCS for X values',
        default='x_train.csv')
    parser.add_argument(
        '--train-file-y',
        type=str,
        help='Training file local or GCS for Y values',
        default='y_train.csv')
    parser.add_argument(
        '--test-file-x',
        type=str,
        help='Test file local or GCS for X values',
        default='x_test.csv')
    parser.add_argument(
        '--test-file-y',
        type=str,
        help='Testfile local or GCS for Y values',
        default='y_test.csv')
    parser.add_argument(
        '--job-id',
        type=str,
        help='Job ID associated with the training job',
        default='job1')
    parser.add_argument(
        '--project-id',
        type=str,
        default='mwe-sanofi-ml-workshop',
        help='The GCP Project ID')
    parser.add_argument(
        '--bucket-name',
        type=str,
        default='sanofi-ml-workshop-black-friday',
        help='The Cloud Storage bucket to be used for process artifacts')
    parser.add_argument(
        '--dataset-id',
        type=str,
        default='black_friday',
        help='The Dataset ID to be used in BigQuery for storing preprocessed data')
    parser.add_argument(
        '--create-data',
        type=bool,
        default=False,
        help='Whether or not to create data for train, test, and validation')
    parser.add_argument(
        '--hp-tune',
        type=bool,
        default=False,
        help='Whether or not to complete hyperparameter tuning')
    parser.add_argument(
        '--num-hp-iterations',
        type=int,
        default=100,
        help='Number of iterations to use for hyperparameter tuning')
    args, _ = parser.parse_known_args()
    train_and_evaluate(args)
