from google.cloud import storage
import pandas as pd
import numpy as np
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project-id',
        type=str,
        default='mwe-demo-ml-workshop',
        help='The GCP Project ID')
    parser.add_argument(
        '--bucket-name',
        type=str,
        default='demo-ml-workshop-chicago-taxi',
        help='The Cloud Storage bucket to be used for process artifacts')
    args, _ = parser.parse_known_args()

    # download the test data
    storage_client = storage.Client(project=args.project_id) 
    bucket = storage_client.get_bucket(args.bucket_name) 
    blob = bucket.blob('x_test.csv')
    x_test = blob.download_as_string()

    # createa a DataFrame
    df = pd.DataFrame(x_test.decode("utf-8").split('\n'))

    # choose a row at random
    which_row = np.random.randint(0, df.shape[0])

    # output as a string
    output_string = '[' + ','.join(list(df.iloc[which_row])) + ']'

    # write string to disk
    json_file = open('input.json', 'w')
    json_file.write(output_string)
    json_file.close()