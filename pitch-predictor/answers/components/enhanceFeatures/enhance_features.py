# libraries
import argparse
from google.cloud import storage
import pandas as pd
import googleapiclient.discovery
import logging

def collectPitch(pitch):
    pitch_types = ['FT','FS','CH','FF','SL','CU','FC','SI','KC','EP','KN','FO']
    
    pitch_label_df = pitch[pitch_types]
    
    pitch_data = pitch.drop(pitch_types,axis=1).values.tolist()
    
    return pitch_data, pitch_label_df


def returnPred(pitch_data):
    service = googleapiclient.discovery.build('ml', 'v1')
    pitch_types = ['FT','FS','CH','FF','SL','CU','FC','SI','KC','EP','KN','FO']
    
    preds_dict = {}
    for pitch_type in pitch_types:
        MODEL_NAME = 'xgboost_' + pitch_type
        name = 'projects/{{ GCP_PROJECT }}/models/{}'.format(MODEL_NAME)

        response = service.projects().predict(
                name=name,
                body={'instances': pitch_data}
            ).execute()

        pred = response['predictions']

        preds_dict[pitch_type] = pred[0]
        
    pred_df = pd.DataFrame.from_dict(preds_dict,orient='index').transpose()
        
    return pred_df


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dummy_1', dest='dummy_1', default='', help='')
    parser.add_argument('--dummy_2', dest='dummy_2', default='', help='')
    parser.add_argument('--dummy_3', dest='dummy_3', default='', help='')
    parser.add_argument('--dummy_4', dest='dummy_4', default='', help='')
    parser.add_argument('--dummy_5', dest='dummy_5', default='', help='')
    parser.add_argument('--dummy_6', dest='dummy_6', default='', help='')
    parser.add_argument('--dummy_7', dest='dummy_7', default='', help='')
    parser.add_argument('--dummy_8', dest='dummy_8', default='', help='')
    parser.add_argument('--dummy_9', dest='dummy_9', default='', help='')
    parser.add_argument('--dummy_10', dest='dummy_10', default='', help='')
    parser.add_argument('--dummy_11', dest='dummy_11', default='', help='')
    parser.add_argument('--dummy_12', dest='dummy_12', default='', help='')

    known_args, _ = parser.parse_known_args(argv)

    #TODO: convert this into a DataFlow job


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

    cols = [] # initiate empy list to collect columns of the eventual DataFrame
    rows = [] # initiate empty list to collect the rows of the eventual DataFrame

    for i in range(0,500):  #len(df)
        # collect the pitch detail from the metrics
        pitch = df.iloc[[i]].reset_index(drop=True) 
        
        # split the pitch detail from the labels
        pitch_data, pitch_label_df = collectPitch(pitch)
        
        # collect predictions
        preds_df = returnPred(pitch_data) 
        
        # join predictions with the label
        new_row_df = preds_df.join(pitch_label_df, lsuffix='_pred') 
        
        # check whether the eventual DataFrame columns need defined
        if len(cols) == 0: 
            cols = list(new_row_df.columns)
        
        # add the combined result to the rows
        new_row = new_row_df.values.tolist()[0]
        rows.append(new_row)
        
    # create a DataFrame from all the rows
    df = pd.DataFrame(rows, columns=cols)

    # write the DataFrame to dish
    df.to_csv('enhanced_features.csv',index=False)

    # upload the results to Cloud Storage
    bucket_name = 'enhanced-features'
    destination_blob_name = 'enhanced_features.csv'

    source_file_name = 'enhanced_features.csv'

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    run()



