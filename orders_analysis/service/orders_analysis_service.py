from abc import ABC, abstractmethod


class OrdersAnalysisService(ABC):
    @abstractmethod
    async def train_model(self):
        pass
