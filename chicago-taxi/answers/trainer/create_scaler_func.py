from google.cloud import storage 
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import trainer.model as model

def create_scaler_func(filenames, CSV_COLUMNS, LABEL_COLUMN, BUCKET_NAME, project_id):
    """Fit MinMaxScaler objects on the dependent and independent variables in the training set

    Args:
        CSV_COLUMNS: column in the training input
        LABEL_COLUMN: label in the training input

    Returns:
        None: writes x_scaler and y_scaler to Cloud Storage
    """

    input_reader = pd.read_csv(
        tf.io.gfile.GFile(filenames[0]),
        names=CSV_COLUMNS,
        chunksize=200000)

    x_scaler = MinMaxScaler()

    # iteratively fit the scalers
    fit_part = 1
    for input_data in input_reader:
        # clean up any rows that contain column headers (in case they were inserted during the sharding and recombining)
        input_data['log_trip_seconds_num'] = input_data['log_trip_seconds'].apply(lambda x: x!='log_trip_seconds')

        input_data = input_data[input_data['log_trip_seconds_num'] == True]
        input_data = input_data.drop(['log_trip_seconds_num'],axis=1)
    
        label = input_data.pop(LABEL_COLUMN).astype(float)
        features, weekday_cat, pickup_census_tract_cat, dropoff_census_tract_cat, pickup_community_area_cat, dropoff_community_area_cat = model.to_numeric_features(input_data)
    
        x_scaler.partial_fit(features.values)
        logging.info('Completed fit part {}'.format(str(fit_part)))
        fit_part +=1
        
    
    # save fitted scalers to Cloud Storage
    joblib.dump(x_scaler, 'x_scaler')
    
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.get_bucket(BUCKET_NAME)
    
    blob = bucket.blob('scalers/x_scaler')
    blob.upload_from_filename('x_scaler')