import random

from random_number.entity.number_boundary import NumberBoundary


class GameNumber:
    def __init__(self):
        self.game_number = random.randint(NumberBoundary.ONE.value, NumberBoundary.TEN.value)

    def getGameNumber(self):
        return self.game_number
