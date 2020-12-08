import apache_beam as beam

class collectStats(beam.DoFn):

    def process(self, row):
        from bs4 import BeautifulSoup
        from google.cloud import storage
        import pandas as pd
        import requests

        pitcher_url = 'http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=' + row.pitcher_id + '&game=' + row.game_id + '/&s_type=&h_size=700&v_size=500'

        pitcher_request = requests.get(pitcher_url)
        pitcher_df = pd.read_html(pitcher_request.text)

        # write the DataFrame to disk
        df = pitcher_df[0]
        source_file_name = row.game_id + '-' + row.pitcher_id + '.csv'
        df.to_csv(source_file_name,index=False)
        
        # upload the file to GCS
        # storage_client = storage.Client()
        # bucket_name = 'mlb-games'
        # destination_blob_name = row.year + '_' + row.month + '_' + row.day + '/' + row.game_id + '/' + row.pitcher_id + '.csv'

        # bucket = storage_client.get_bucket(bucket_name)
        # blob = bucket.blob(destination_blob_name)
        # blob.upload_from_filename(source_file_name)

        # yield one row as a time as a dictionary
        df = df.applymap(str) # convert all fields to strings
        for i in range(0, len(df)):
            out_dict = df.iloc[i].to_dict()
            # omit slash from the game_id
            out_dict['gid'] = out_dict['gid'].strip('/')
            # add pitcher_id to dict
            out_dict['pitcher_id'] = row.pitcher_id

            yield out_dict

