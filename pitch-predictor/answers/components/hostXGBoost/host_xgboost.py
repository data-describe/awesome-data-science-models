# libraries
import argparse
import datetime
import shutil
from google.cloud import storage
import logging
import pandas as pd
import os
import subprocess




def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pitch_type', dest='pitch_type', default='SI', help='Select the pitch type to evaluate')

    known_args, _ = parser.parse_known_args(argv)

    # define the pitch type 
    pitch_type = known_args.pitch_type

    # define some contants for AI Platform
    bucket_name = '{{ GCP_PROJECT }}-pitch-data'
    MODEL_DIR = f'gs://{bucket_name}/' + pitch_type + '/'
    MODEL_NAME = 'xgboost_' + pitch_type
    VERSION_NAME = datetime.datetime.now().strftime(pitch_type + '_%Y%m%d%M')    
    FRAMEWORK = 'XGBOOST'

    # check to confirm whether the model has already been created
    proc = subprocess.Popen([shutil.which('gcloud'),'ai-platform','models','list'], stdout=subprocess.PIPE)
    output = proc.stdout.read().decode("utf-8")
    if MODEL_NAME not in output:
        # create a model in AI Platform
        subprocess.check_call([shutil.which('gcloud'),'beta', 'ai-platform','models','create',MODEL_NAME,'--regions','us-central1', "--enable-logging", "--enable-console-logging"])

    # create a new version with our trained model
    subprocess.check_call([shutil.which('gcloud'),'ai-platform','versions','create',VERSION_NAME,'--model',MODEL_NAME,'--origin',MODEL_DIR,'--runtime-version','1.15','--framework',FRAMEWORK,'--python-version','3.7'])

    # set the newest version to the default version
    subprocess.check_call([shutil.which('gcloud'),'ai-platform','versions','set-default',VERSION_NAME,'--model',MODEL_NAME])


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    run()



