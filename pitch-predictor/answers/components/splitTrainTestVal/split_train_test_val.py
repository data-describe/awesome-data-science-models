# libraries
import argparse
from google.cloud import storage
import logging
import numpy as np
import pandas as pd


def run(argv=None): #
    parser = argparse.ArgumentParser()
    parser.add_argument('--pitch_type', dest='pitch_type', default='SI', help='Select the pitch type to evaluate')

    known_args, _ = parser.parse_known_args(argv)

    # define the pitch type 
    pitch_type = known_args.pitch_type

    # download the raw metrics data
    bucket_name = 'raw-pitch-data'
    source_blob_name = 'metrics.csv'
    destination_file_name = 'metrics.csv'

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    # load raw data into DataFrame
    df = pd.read_csv('metrics.csv')
    # omit unnecessary pitch types
    all_pitchtypes = ['FT','FS','CH','FF','SL','CU','FC','SI','KC','EP','KN','FO']            
    all_pitchtypes.remove(pitch_type)
    df = df.drop(all_pitchtypes, axis=1)

    # split the into true and false DataFrames
    df_true = df[df[pitch_type] == 1]
    df_false = df[df[pitch_type] == 0]

    # split both into train test and validation sets
    np.random.seed(1)

        # split the true DataFrame
    msk = np.random.rand(len(df_true)) < 0.7
    df_true_train = df_true[msk].reset_index(drop=True)
    df_true_test_val = df_true[~msk].reset_index(drop=True)

    msk = np.random.rand(len(df_true_test_val)) < 0.5
    df_true_test = df_true_test_val[msk].reset_index(drop=True)
    df_true_val = df_true_test_val[~msk].reset_index(drop=True)

    assert df_true.shape[0] == df_true_val.shape[0] + df_true_train.shape[0] + df_true_test.shape[0]

        # split the false DataFrame
    msk = np.random.rand(len(df_false)) < 0.7
    df_false_train = df_false[msk].reset_index(drop=True)
    df_false_test_val = df_false[~msk].reset_index(drop=True)

    msk = np.random.rand(len(df_false_test_val)) < 0.5
    df_false_test = df_false_test_val[msk].reset_index(drop=True)
    df_false_val = df_false_test_val[~msk].reset_index(drop=True)

    assert df_false.shape[0] == df_false_val.shape[0] + df_false_train.shape[0] + df_false_test.shape[0]

    # combine the true and false DataFrames
    df_train = pd.concat([df_false_train, df_true_train])
    df_test = pd.concat([df_false_test, df_true_test])
    df_val = pd.concat([df_false_val, df_true_val])

    # shuffle the dataframes
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)

    # write results to disk
    df_train.to_csv('train.csv',index=False)
    df_test.to_csv('test.csv',index=False)
    df_val.to_csv('val.csv',index=False)

    # push results to GCS
    bucket_name = 'train-test-val'
    bucket = storage_client.get_bucket(bucket_name)

        # train
    destination_blob_name = pitch_type + '/' + 'train.csv'
    source_file_name = 'train.csv'


    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

        # test
    destination_blob_name = pitch_type + '/' + 'test.csv'
    source_file_name = 'test.csv'


    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

        # val
    destination_blob_name = pitch_type + '/' + 'val.csv'
    source_file_name = 'val.csv'


    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    run()



