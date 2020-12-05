import apache_beam as beam

class collectGames(beam.DoFn):

    def process(self, row):
        from bs4 import BeautifulSoup
        from google.cloud import storage
        import requests
        import re

        game_url = f'http://www.brooksbaseball.net/tabs.php?player={row.pitcher_id}&var=gl'
        game_request = requests.get(game_url)
        game_soup = BeautifulSoup(game_request.text, features="html.parser")

        for link in game_soup.find_all('a'):
            match = re.match(r"([A-Z]{3})@([A-Z]{3})\s+\((\d+)\/(\d+)\/(\d+)\)", link.text)
            if match:
                vals = match.groups()
                if str(row.year)[-2:] == vals[4]:
                    game_id = f"gid_{row.year}_{vals[2].zfill(2)}_{vals[3].zfill(2)}_{vals[0].lower()}mlb_{vals[1].lower()}mlb_1"
                    row.game_id = game_id
                    yield row