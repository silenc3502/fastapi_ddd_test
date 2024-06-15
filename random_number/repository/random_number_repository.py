from abc import ABC, abstractmethod


class RandomNumberRepository(ABC):
    @abstractmethod
    def draw(self):
        pass
    