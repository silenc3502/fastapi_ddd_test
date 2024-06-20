import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from random_forest.repository.random_forest_repository import RandomForestRepository

from imblearn.over_sampling import SMOTE


class RandomForestRepositoryImpl(RandomForestRepository):

    def flightCategoricalVariableEncoding(self, data):
        print("flightCategoricalVariableEncoding()")
        # 범주형 변수 인코딩
        label_encoders = {}
        categorical_columns = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])
        return data, label_encoders

    def splitTrainTestSet(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def applySmote(self, X_train, y_train):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        return X_resampled, y_resampled

    def train(self, X_train, y_train):
        print("train()")
        # # 훈련/테스트 데이터 분할
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #
        # # SMOTE를 이용한 데이터 불균형 문제 해결
        # smote = SMOTE(random_state=42)
        # X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # 랜덤 포레스트 모델을 학습
        randomForestModel = RandomForestClassifier(n_estimators=100, random_state=42)
        randomForestModel.fit(X_train, y_train)

        return randomForestModel
        # y_pred = rf_model_smote.predict(X_test)
        #
        # return X_train, X_test, y_train, y_test, y_pred

    def predict(self, randomForestModel, X_test):
        y_pred = randomForestModel.predict(X_test)

        return y_pred

    def evaluate(self, y_test, y_pred):
        print("predict()")
        # 평가
        accuracy = accuracy_score(y_test, y_pred)
        print("pass accuracy")
        report = classification_report(y_test, y_pred)
        print("pass classification")
        confusionMatrix = confusion_matrix(y_test, y_pred)
        print("pass confusion matrix")

        return accuracy, report, confusionMatrix

    # def predict(self, y_test, y_pred_smote):
    #     print("predict()")
    #     # 평가
    #     accuracy_smote = accuracy_score(y_test, y_pred_smote)
    #     print("pass accuracy")
    #     report_smote = classification_report(y_test, y_pred_smote)
    #     print("pass classification")
    #     cm = confusion_matrix(y_test, y_pred_smote)
    #     print("pass confusion matrix")
    #
    #     return accuracy_smote, report_smote, cm

    # def predict(self, y_test, y_pred_smote):
    #     print("predict()")
    #     # 평가
    #     accuracy_smote = accuracy_score(y_test, y_pred_smote)
    #     print("pass accuracy")
    #
    #     report_dict = classification_report(y_test, y_pred_smote, output_dict=True)
    #     report_json = json.dumps(report_dict)
    #     print("pass classification")
    #
    #     cm = confusion_matrix(y_test, y_pred_smote).tolist()
    #     print("pass confusion matrix")
    #
    #     return accuracy_smote, report_json, cm
