# libraries
import argparse
import datetime
from google.cloud import storage
import googleapiclient.discovery
import logging
import numpy as np
import pandas as pd
import pandas_gbq

from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score, roc_curve,precision_recall_curve, auc, confusion_matrix
#import matplotlib.pyplot as plt



def returnPreds(preds, threshold):
    """
    Return predicted binary values from prediction probabilities based on a given threshold

    Inputs:
        - preds: prediction probabilities
        - threshold: threshold for positive prediction value

    Output:
        - predicted_labels list of binary predicted labels based on threshold
    """            
    predicted_labels = []
    for pred in preds:
        if pred >= threshold:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    return predicted_labels

def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pitch_type', dest='pitch_type', default='SI', help='Select the pitch type to evaluate')

    known_args, _ = parser.parse_known_args(argv)

    # define the pitch type 
    pitch_type = known_args.pitch_type


    # download the  data
    storage_client = storage.Client()
    bucket_name = 'train-test-val'
        # test
    source_blob_name = pitch_type + '/test.csv'
    destination_file_name = 'test.csv'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    df_test = pd.read_csv('test.csv')
        # threshold
    bucket_name = 'thresholds'
    source_blob_name = pitch_type + '/threshold.txt'
    destination_file_name = 'threshold.txt'


    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    threshold = eval(blob.download_as_string())

    # define model name
    MODEL_NAME = 'xgboost_' + pitch_type
    # define the service
    service = googleapiclient.discovery.build('ml', 'v1')
    # define the model
    name = 'projects/ross-kubeflow/models/{}'.format(MODEL_NAME)

    # define validation data and labels
    test_labels = df_test[pitch_type].values.tolist()
    test_data = df_test.drop([pitch_type],axis=1).values.tolist()

    # collect validation predictions

    # split the validation into chunks so that they can be passed to the online prediction
    # Note -- could also use batch predictions

    test_data_parts = list(divide_chunks(test_data, 200))

    test_preds = []
    for part in test_data_parts: 
        response = service.projects().predict(
            name=name,
            body={'instances': part}
        ).execute()

        test_preds.extend(response['predictions'])

    # add predictions to Test DataFrame
    df_test['pred_score'] = test_preds


    # turn our prediction probabilities from the test set into actual predictions with this threshold
    predicted_test_labels = returnPreds(test_preds, threshold)


    # calculate accuracy metrics
    precision = precision_score(test_labels, predicted_test_labels)
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    recall = recall_score(test_labels, predicted_test_labels)
    f1 = f1_score(test_labels, predicted_test_labels)

    # upload model results to BigQuery
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame([['xgboost',pitch_type,precision,accuracy,recall,f1,threshold,today]],columns=['model_type','pitch_type','precision','accuracy','recall','f1','threshold','date'])
    df.to_gbq(destination_table='baseball.models',project_id='ross-kubeflow',if_exists='append')


    ''' COMMENTING PLOTTING FUNCTIONS
    # plot the ROC and PR curves
    plotPRandROC(test_labels, test_preds)

    # plot probability distributions for positive and negative samples
    positive_probs = df_test[df_test[pitch_type] == 1]['pred_score'].values
    negative_probs = df_test[df_test[pitch_type] == 0]['pred_score'].values

    plotProbDist(positive_probs, negative_probs)

    # Plot confusion matrix
    plotConfusionMatrix(test_labels, predicted_test_labels, classes=np.asarray([0,1]))
    '''

    # write fake file to pass as output
    f = open('/root/dummy.txt', 'w')
    f.write('dummy text')
    f.close()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    run()



