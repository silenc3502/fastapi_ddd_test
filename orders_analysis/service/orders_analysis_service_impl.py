import os
import aiofiles
import joblib
import pandas as pd

from orders_analysis.repository.orders_analysis_repository_impl import OrdersAnalysisRepositoryImpl
from orders_analysis.service.orders_analysis_service import OrdersAnalysisService


class OrdersAnalysisServiceImpl(OrdersAnalysisService):
    def __init__(self):
        self.ordersAnalysisRepository = OrdersAnalysisRepositoryImpl()

    async def readExcel(self):
        currentDirectory = os.getcwd()
        print(f"current directory: {currentDirectory}")

        filePath = os.path.join(
            currentDirectory, "assets", "orders_data_after_drop_duplication.xlsx")

        try:
            df = pd.read_excel(filePath)
            return df
        except FileNotFoundError:
            print(f"File not found: {filePath}")

    async def train_model(self):
        dataFrame = await self.readExcel()
        X_scaled, y, scaler = await self.ordersAnalysisRepository.prepareData(dataFrame)

        joblib.dump(scaler, "scaler.pkl")

        X_train, X_test, y_train, y_test = (
            await self.ordersAnalysisRepository.splitTrainTestData(X_scaled, y))

        num_models = 10
        modelList = []

        for i in range(num_models):
            print(f"Training model {i + 1}/{num_models}")

            model = await self.ordersAnalysisRepository.createModel()
            await self.ordersAnalysisRepository.fitModel(model, X_train, y_train)
            model.save(f"model_{i + 1}.h5")
            modelList.append(model)

        return f"Trained {num_models} models successfully."
