from google.cloud import storage
import joblib
import json
import sys
import numpy as np
from os import path
import random

from trainer import model

PROJECT_ID = 'mwpmltr'
BUCKET_NAME = 'ross-keras'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__=='__main__':
      # download the scaler
    if not path.exists('x_scaler'):
        print('Downloading scaler')
        storage_client = storage.Client(project='mwpmltr')
        bucket = storage_client.get_bucket('ross-keras')
        blob = bucket.blob('scalers/x_scaler')
        blob.download_to_filename('x_scaler')
        print('Downloaded scaler')

    x_scaler = joblib.load('x_scaler')

    gen = model.generator_input(['gs://ross-keras/data/full_test_results.csv'], chunk_size=5000, project_id=PROJECT_ID, bucket_name=BUCKET_NAME, x_scaler=x_scaler, batch_size=1)
    

    for i in range(1, random.randint(1,100)):
        sample = gen.__next__()

    input_sample = {}
    input_sample['input'] = sample[0]
    
    
    print('Produced sample with label {} seconds.'.format(str(int(round(np.exp(sample[1]))))))

    with open('input_sample.json', 'w') as outfile:
        json.dump(input_sample, outfile, cls=NumpyEncoder)