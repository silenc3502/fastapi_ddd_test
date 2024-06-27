from abc import ABC, abstractmethod


class OrdersAnalysisRepository(ABC):
    @abstractmethod
    async def prepareData(self, dataFrame):
        pass

    @abstractmethod
    async def splitTrainTestData(self, X_scaled, y):
        pass

    @abstractmethod
    async def createModel(self):
        pass

    @abstractmethod
    async def fitModel(self, model, X_train, y_train):
        pass
