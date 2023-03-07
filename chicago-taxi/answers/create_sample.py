from google.cloud import storage
import joblib
import json
import sys
import numpy as np
from os import path
import random
import argparse

from trainer import model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project-id',
        type=str,
        default='mwe-sanofi-ml-workshop',
        help='The GCP Project ID')
    parser.add_argument(
        '--bucket-name',
        type=str,
        default='sanofi-ml-workshop-chicago-taxi-demo',
        help='The Cloud Storage bucket to be used for process artifacts')
    args, _ = parser.parse_known_args()
    
      # download the scaler
    if not path.exists('x_scaler'):
        print('Downloading scaler')
        storage_client = storage.Client(project=args.project_id)
        bucket = storage_client.get_bucket(args.bucket_name)
        blob = bucket.blob('scalers/x_scaler')
        blob.download_to_filename('x_scaler')
        print('Downloaded scaler')

    x_scaler = joblib.load('x_scaler')

    gen = model.generator_input(['gs://{}/data/full_test_results.csv'.format(args.bucket_name)], chunk_size=5000, project_id=args.bucket_name, bucket_name=args.bucket_name, x_scaler=x_scaler, batch_size=1)
    

    for i in range(1, random.randint(1,100)):
        sample = gen.__next__()

    input_sample = {}
    input_sample['input'] = sample[0]
    
    
    print('Produced sample with label {} seconds.'.format(str(int(round(np.exp(sample[1]))))))

    with open('input_sample.json', 'w') as outfile:
        json.dump(input_sample, outfile, cls=NumpyEncoder)