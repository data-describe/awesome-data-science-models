import argparse
from google.cloud import storage
import joblib
import json
import sys
import numpy as np
from os import path
import random
import shlex
import subprocess

from trainer import model

def mean_absolute_percentage_error(y_true, y_pred): 

    return int(round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--num-samples',
      type=int,
      help='Number of test samples to use for calculating RMSE',
      default=5)
    parser.add_argument(
      '--model',
      type=str,
      help='Model used to make predictions',
      default='keras_wnd_model')
    parser.add_argument(
      '--version',
      type=str,
      help='Model version used to make predictions',
      default='v1')
    parser.add_argument(
      '--project-id',
      type=str,
      help='GCP project where the model is hosted on AI Platform',
      default='mwpmltr')
    parser.add_argument(
      '--bucket-name',
      type=str,
      help='GCP bucket where model artifacts are stored',
      default='ross-keras')
    
    args, _ = parser.parse_known_args()
    
    # download the scaler
    if not path.exists('x_scaler'):
        print('Downloading scaler')
        storage_client = storage.Client(project='mwpmltr')
        bucket = storage_client.get_bucket('ross-keras')
        blob = bucket.blob('scalers/x_scaler')
        blob.download_to_filename('x_scaler')
        print('Downloaded scaler')

    x_scaler = joblib.load('x_scaler')

    gen = model.generator_input(['gs://ross-keras/data/full_test_results.csv'], chunk_size=5000, project_id=args.project_id, bucket_name=args.bucket_name, x_scaler=x_scaler, batch_size=1)
    
    actuals = []
    preds = []
    for i in range(0, args.num_samples):
        sample = gen.__next__()

        input_sample = {}
        input_sample['input'] = sample[0]
    
    
        
        actual = int(round(np.exp(sample[1])))
        actuals.append(actual)

        with open('input_sample.json', 'w') as outfile:
            json.dump(input_sample, outfile, cls=NumpyEncoder)
            
        command = "gcloud ai-platform predict --model={} --version={} --json-instances=input_sample.json".format(args.model, args.version)
        output = subprocess.check_output(shlex.split(command))
        output_json = json.loads(output)
        
        pred = int(round(eval(output_json['predictions'])))
        preds.append(pred)
        
        print('Returned sample with label {} and prediction {}.'.format(str(actual), str(pred)))
    
    mape = mean_absolute_percentage_error(np.asarray(actuals), np.asarray(preds)) 
    
    print('MAPE across {} test samples is {}%.'.format(str(args.num_samples), str(mape)))
    
    