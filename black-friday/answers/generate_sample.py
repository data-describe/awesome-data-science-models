from google.cloud import storage
import pandas as pd
import numpy as np

project_id = 'mwe-sanofi-ml-workshop'
bucket_name = 'sanofi-ml-workshop-black-friday'

# download the test data
storage_client = storage.Client(project=project_id) 
bucket = storage_client.get_bucket(bucket_name) 
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