from abc import ABC, abstractmethod


class WordCloudRepository(ABC):
    @abstractmethod
    def create(self, text):
        pass
    