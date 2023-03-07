from google.cloud import storage
import pandas as pd

def collectSampleEnhanced():
    # download the  data
    storage_client = storage.Client()
    bucket_name = 'enhanced-features'
        # val
    source_blob_name = 'enhanced_features.csv'
    destination_file_name = 'enhanced_features.csv'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    df = pd.read_csv('enhanced_features.csv')
    
    return df