from google.cloud import bigquery
from google.cloud import storage   
import logging
import subprocess

def create_data_func(data_part, project_id, bucket_name, dataset_id):
    # create a temporary table with the results and then export to cloud storage
    client = bigquery.Client(project=project_id)

    results_table = '{}_results'.format(data_part)

    # delete the existing table if necessary
    try:
        client.delete_table('{}.{}'.format(dataset_id, results_table))
    except: # table not found
        pass

    job_config = bigquery.QueryJobConfig()
    # Set the destination table
    table_ref = client.dataset(dataset_id).table(results_table)
    job_config.destination = table_ref
    sql = """
    SELECT  
     LOG(trip_seconds) AS log_trip_seconds
    ,ST_DISTANCE(ST_GEOGPOINT(pickup_longitude, pickup_latitude), ST_GEOGPOINT(dropoff_longitude, dropoff_latitude)) AS distance
    ,EXTRACT(HOUR FROM trip_start_timestamp) AS hour_start
    ,EXTRACT(MONTH FROM trip_start_timestamp) AS month_start
    ,FORMAT_TIMESTAMP('%a', trip_start_timestamp) AS weekday
    ,IFNULL(pickup_census_tract,9999) AS pickup_census_tract
    ,IFNULL(dropoff_census_tract,9999) AS dropoff_census_tract
    ,IFNULL(pickup_community_area,9999) AS pickup_community_area
    ,IFNULL(dropoff_community_area,9999) AS dropoff_community_area
    ,pickup_latitude
    ,pickup_longitude
    ,dropoff_latitude
    ,dropoff_longitude
    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips` 
    WHERE ST_DISTANCE(ST_GEOGPOINT(pickup_longitude, pickup_latitude), ST_GEOGPOINT(dropoff_longitude, dropoff_latitude)) > 0
    AND trip_seconds <= 60*60*2
    AND trip_seconds >= 60
    """
    
    if data_part == 'train':
        sql += "AND MOD(FARM_FINGERPRINT(unique_key), 5) != 0 -- 80% to train"
    elif data_part == 'val':
        sql += "AND MOD(FARM_FINGERPRINT(unique_key), 5) = 0\nAND MOD(FARM_FINGERPRINT(unique_key), 10) != 0 -- 10% val"
    elif data_part == 'test':
        sql += "AND MOD(FARM_FINGERPRINT(unique_key), 5) = 0\nAND MOD(FARM_FINGERPRINT(unique_key), 10) = 0 -- 10% test"

    # Start the query, passing in the extra configuration.
    query_job = client.query(
      sql,
      # Location must match that of the dataset(s) referenced in the query
      # and of the destination table.
      location='US',
      job_config=job_config)  # API request - starts the query

    query_job.result()  # Waits for the query to finish

    logging.info('Query results loaded to table {}'.format(table_ref.path))

    # load table to Cloud Storage
        # delete any files currently present
    subprocess.run(['gsutil', '-m' ,'rm', 'gs://{}/data/{}/**'.format(bucket_name, data_part)])
    logging.info('Existing files removed from gs://{}/data/{}/'.format(bucket_name, data_part))
        # extract tabe to cloud storage
    extract_job = client.extract_table(source='{}.{}.{}'.format(project_id, dataset_id, results_table), destination_uris='gs://{}/data/{}/{}_results_*.csv'.format(bucket_name, data_part, data_part))
    extract_job.result()  # Waits for the extract to finish
    logging.info('Query results uploaded to GCS {}'.format('gs://{}/data/{}/{}_results_*.csv'.format(bucket_name, data_part, data_part)))
        # combine the shards into a single file
    subprocess.run(['gsutil', 'compose', 'gs://{}/data/{}/{}_results_*.csv'.format(bucket_name, data_part, data_part), 'gs://{}/data/full_{}_results.csv'.format(bucket_name, data_part)])
    logging.info('Query results combined in GCS {}'.format('gs://{}/data/full_{}_results.csv'.format(bucket_name, data_part)))
            # delete the shards
    subprocess.run(['gsutil', '-m', 'rm', 'gs://{}/data/{}/**'.format(bucket_name, data_part)])
    logging.info('Temp files removed from gs://{}/data/{}/'.format(bucket_name, data_part))