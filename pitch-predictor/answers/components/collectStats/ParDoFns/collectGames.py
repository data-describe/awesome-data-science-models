import apache_beam as beam

class collectGames(beam.DoFn):

    def process(self, row):
        from bs4 import BeautifulSoup
        from google.cloud import storage
        import requests

        # check to confirm whether the date has already been processed
        date_folder = row.year + '_' + row.month + '_' + row.day
        storage_client = storage.Client()
        bucket_name = 'mlb-games'
        bucket = storage_client.get_bucket(bucket_name)
        dates_processed = []
        blobs = bucket.list_blobs()
        for blob in blobs:
            dates_processed.append(blob.name.split('/')[0])

        if date_folder not in dates_processed:

            day_url = 'http://www.brooksbaseball.net/pfxVB/pfx.php?month=' + row.month + '&day=' + row.day + '&year=' + row.year

            day_request = requests.get(day_url)
            day_soup = BeautifulSoup(day_request.text, features="lxml")

            for i in range(0, len(day_soup.find_all('select'))):
                if day_soup.find_all('select')[i].get('name') == 'game': # scroll through the day's games
                    for game_opt in day_soup.find_all('select')[i].find_all('option'):
                        game_id = game_opt.get('value').strip('/')
                        row.game_id = game_id

                        yield row