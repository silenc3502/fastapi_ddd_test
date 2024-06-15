from random_number.repository.random_number_repository_impl import RandomNumberRepositoryImpl
from random_number.service.random_number_service import RandomNumberService


class RandomNumberServiceImpl(RandomNumberService):
    def __init__(self):
        self.randomNumberRepository = RandomNumberRepositoryImpl()

    async def drawRandomNumber(self):
        randomNumber = await self.randomNumberRepository.draw()
        return randomNumber
