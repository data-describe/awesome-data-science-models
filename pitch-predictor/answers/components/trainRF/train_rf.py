# libraries
import argparse
import logging
from google.cloud import storage
import joblib
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def run(argv=None):
    # download the enhanced features
    bucket_name = 'enhanced-features'
    source_blob_name = 'enhanced_features.csv'
    destination_file_name = 'enhanced_features.csv'

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    # load the features
    df = pd.read_csv('enhanced_features.csv')

    # collect the max index for each of the labels
    pitch_types = ['FT','FS','CH','FF','SL','CU','FC','SI','KC','EP','KN','FO']

    labels = []
    for i in range(0, len(df)):
        values = df.iloc[[i]][pitch_types].values.tolist()[0]
        max_index = values.index(max(values))
        labels.append(max_index)

    # drop the initial pitchtype columns
    df = df.drop(pitch_types, axis=1)

    # define data and labels
    y = np.asarray(labels)
    X = df.values
    
    # train the RF classifier
    clf = RandomForestClassifier()
    cv = KFold(10) 
    clf = GridSearchCV(clf, {'n_estimators': [10, 100, 1000]}, n_jobs=-1, cv=cv)
    clf.fit(X, y)

    # save the model to disk
    '''
    joblib.dump(clf, 'model.joblib')
    '''
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(clf, model_file)

    # push the model to cloud storage
    bucket_name = 'rf-model'
    destination_blob_name = 'model.pkl'
    source_file_name = 'model.pkl'

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    run()



