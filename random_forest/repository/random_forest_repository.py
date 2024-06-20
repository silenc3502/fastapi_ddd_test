from abc import ABC, abstractmethod


class RandomForestRepository(ABC):
    @abstractmethod
    def flightCategoricalVariableEncoding(self, data):
        pass

    @abstractmethod
    def splitTrainTestSet(self, X, y):
        pass

    @abstractmethod
    def applySmote(self, X_train, y_train):
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, randomForestModel, X_test):
        pass

    @abstractmethod
    def evaluate(self, y_test, y_pred):
        pass
