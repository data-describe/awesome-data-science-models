# libraries
import argparse
import datetime
from google.cloud import storage
import googleapiclient.discovery
import logging
import numpy as np
import pandas as pd
import os

from sklearn.metrics import f1_score


def returnThreshold(preds, labels):
    """
    Determine a threshold to set positive values in order to maximize F1 score

    Inputs:
        - preds: predicted probabilities
        - labels: actual labels

    Output:
        - threshold for positive values that maximizes F1 score
    """
    best_f1 = 0
    best_threshold = 0
    for i in np.arange(0,1,.001):
        predicted_labels = []
        for pred in preds:
            if pred >= i:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
        
        f1 = f1_score(labels, predicted_labels)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = i
            
    return str(best_threshold)

def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]


def run(argv=None):
    GCP_PROJECT = os.getenv("GCP_PROJECT")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pitch_type', dest='pitch_type', default='SI', help='Select the pitch type to evaluate')

    known_args, _ = parser.parse_known_args(argv)

    # define the pitch type 
    pitch_type = known_args.pitch_type


    # download the  data
    storage_client = storage.Client()
    bucket_name = f'{GCP_PROJECT}-pitch-data'
        # val
    source_blob_name = pitch_type + '/val.csv'
    destination_file_name = 'val.csv'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    df_val = pd.read_csv('val.csv')

    # define model name
    MODEL_NAME = 'xgboost_' + pitch_type
    # define the service
    service = googleapiclient.discovery.build('ml', 'v1')
    # define the model
    name = f'projects/{GCP_PROJECT}/models/{MODEL_NAME}'

    # define validation data and labels
    val_labels = df_val[pitch_type].values.tolist()
    val_data = df_val.drop([pitch_type],axis=1).values.tolist()

    # collect validation predictions

    # split the validation into chunks so that they can be passed to the online prediction
    # Note -- could also use batch predictions

    val_data_parts = list(divide_chunks(val_data, 200))

    val_preds = []
    for part in val_data_parts: 
        response = service.projects().predict(
            name=name,
            body={'instances': part}
        ).execute()

        val_preds.extend(response['predictions'])

    # find the threshold that maximizes F1
    threshold = returnThreshold(val_preds, val_labels)

    # upload the threshold value to GCS
    destination_blob_name = pitch_type + '/threshold.txt'


    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(threshold)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    run()



