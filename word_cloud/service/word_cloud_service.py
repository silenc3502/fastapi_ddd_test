from abc import ABC, abstractmethod


class WordCloudService(ABC):
    @abstractmethod
    def generateWordCloud(self):
        pass
    