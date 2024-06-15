from random_number.entity.game_number import GameNumber
from random_number.repository.random_number_repository import RandomNumberRepository


class RandomNumberRepositoryImpl(RandomNumberRepository):

    async def draw(self):
        gameNumber = GameNumber()
        randomNumber = gameNumber.getGameNumber()
        return randomNumber
