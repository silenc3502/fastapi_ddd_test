from random_forest.repository.random_forest_repository_impl import RandomForestRepositoryImpl
from random_forest.service.random_forest_service import RandomForestService

import pandas as pd
import os

from random_forest.service.response_form.random_forest_response_form import RandomForestResponseForm


class RandomForestServiceImpl(RandomForestService):
    def __init__(self):
        self.__randomForestRepositoryImpl = RandomForestRepositoryImpl()

    def readCsv(self):
        # 현재 작업 디렉토리(워킹 디렉토리) 출력
        currentDirectory = os.getcwd()
        print("현재 작업 디렉토리:", currentDirectory)

        filePath = os.path.join(currentDirectory, 'assets', 'customer_booking.csv')
        dataFrame = pd.read_csv(filePath, encoding='latin1')
        return dataFrame

    def featureTargetVariableDefinition(self, data):
        print("featureTargetVariableDefinition()")
        # 특징과 타겟 변수 정의
        X = data.drop('booking_complete', axis=1)
        y = data['booking_complete']
        return X, y

    def randomForestAnalysis(self):
        print("randomForestAnalysis()")
        data = self.readCsv()
        data_encoded, label_encoders = self.__randomForestRepositoryImpl.flightCategoricalVariableEncoding(data)
        X, y = self.featureTargetVariableDefinition(data_encoded)

        X_train, X_test, y_train, y_test = self.__randomForestRepositoryImpl.splitTrainTestSet(X, y)
        randomForestModel = self.__randomForestRepositoryImpl.train(X_train, y_train)
        y_pred = self.__randomForestRepositoryImpl.predict(randomForestModel, X_test)
        accuracy, report, confusionMatrix = self.__randomForestRepositoryImpl.evaluate(y_test, y_pred)

        X_resampled, y_resampled = self.__randomForestRepositoryImpl.applySmote(X_train, y_train)
        smoteRandomForestModel = self.__randomForestRepositoryImpl.train(X_resampled, y_resampled)
        y_pred_smote = self.__randomForestRepositoryImpl.predict(smoteRandomForestModel, X_test)
        smoteAccuracy, smoteReport, smoteConfusionMatrix = self.__randomForestRepositoryImpl.evaluate(y_test, y_pred_smote)

        # accuracy, report, confusionMatrix, smoteAccuracy, smoteReport, smoteConfusionMatrix
        # y_test, y_pred_smote, y_pred, data 같은 것들을 뭉쳐서 리턴
        # cm_before_smote, cm_after_smote, y_test, y_pred_before_smote, y_pred_smote, data
        return RandomForestResponseForm.createForm(
            confusionMatrix, smoteConfusionMatrix,
            y_test, y_pred, y_pred_smote, data)

