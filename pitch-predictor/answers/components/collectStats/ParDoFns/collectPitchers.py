import apache_beam as beam

class collectPitchers(beam.DoFn):

    def process(self, row):
        import apache_beam as beam
        import re
        from bs4 import BeautifulSoup
        import requests

        mlb_url = f'https://www.mlb.com/stats/pitching/{row.year}?page='
        for page in range(1, 5):
            mlb_request = requests.get(mlb_url + str(page))
            mlb_soup = BeautifulSoup(mlb_request.text, features="html.parser")

            id_links = mlb_soup.find_all('a', {'class': 'bui-link'})
            for link in id_links:
                if re.match(r"/player/\d+", link['href']):
                    pitcher_id = link['href'].split("/")[-1]
                    row.pitcher_id = pitcher_id
                    yield row
