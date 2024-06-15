from abc import ABC, abstractmethod


class RandomNumberService(ABC):
    @abstractmethod
    async def drawRandomNumber(self):
        pass
