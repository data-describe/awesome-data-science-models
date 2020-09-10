import apache_beam as beam

class collectPitchers(beam.DoFn):

    def process(self, row):
        from bs4 import BeautifulSoup
        import requests

        game_url = 'http://www.brooksbaseball.net/pfxVB/pfx.php?month=' + row.month + '&day=' + row.day + '&year=' + row.year + '&game=' + row.game_id + '%2F&prevDate=827&league=mlb'
        game_request = requests.get(game_url)
        game_soup = BeautifulSoup(game_request.text, features="lxml")



        for j in range(0, len(game_soup.find_all('select'))):
            if game_soup.find_all('select')[j].get('name') == 'pitchSel': # scroll through the game's pitchers
                for pitcher_opt in game_soup.find_all('select')[j].find_all('option'):
                    pitcher_id = pitcher_opt.get('value')

                    row.pitcher_id = pitcher_id

                    yield row
