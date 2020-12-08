# describe class to represent gamedays
class GameDay:
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year
        self.game_id = None
        self.pitcher_id = None
