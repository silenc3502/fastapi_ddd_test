from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

import numpy as np
import joblib
import tensorflow as tf
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

irisAnalysisRouter = APIRouter()

# 모델과 스케일러 파일 경로
MODEL_PATH = 'iris_model.h5'
SCALER_PATH = 'iris_scaler.pkl'

@irisAnalysisRouter.post("/train")
def train_model():
    # Iris 데이터셋 로드 및 모델 훈련
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 모델 정의
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 학습
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    # 모델과 스케일러를 파일로 저장
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return {"message": "Model and scaler trained and saved."}

@irisAnalysisRouter.post("/predict")
def predict(iris: IrisRequest):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=400, detail="Model and scaler not found. Train the model first.")

    # 저장된 모델과 스케일러 로드
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    data = scaler.transform(data)
    prediction = model.predict(data)
    predicted_class = np.argmax(prediction, axis=1)
    return {"prediction": int(predicted_class[0])}
