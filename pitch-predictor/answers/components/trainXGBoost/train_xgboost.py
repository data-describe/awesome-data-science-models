# libraries
import argparse
from google.cloud import storage
import logging
import pandas as pd
import os

import xgboost as xgb



def run(argv=None):
    GCP_PROJECT = os.getenv("GCP_PROJECT")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pitch_type', dest='pitch_type', default='SI', help='Select the pitch type to evaluate')

    known_args, _ = parser.parse_known_args(argv)

    # define the pitch type 
    pitch_type = known_args.pitch_type


    # download the  data
    storage_client = storage.Client()
        # train
    bucket_name = f'{GCP_PROJECT}-pitch-data'
    source_blob_name = pitch_type + '/train.csv'
    destination_file_name = 'train.csv'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    df_train = pd.read_csv('train.csv')
        # hyperparameters
    source_blob_name = pitch_type + '/params.json'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    params = eval(blob.download_as_string())


    # prepare data for the xgboost model
        # training
    train_labels = df_train[pitch_type] == 1
    train_features = df_train.drop(pitch_type, axis=1)

    # train a model with the optimized hyperparameters
    model = xgb.XGBClassifier(**params)
    trained_model = model.fit(train_features, train_labels)

    # save trained model to disk
    model_filename = 'model.bst'
    trained_model.save_model(model_filename)

    # upload model to GCS
    destination_blob_name = pitch_type + '/model.bst'
    source_file_name = 'model.bst'

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    run()



