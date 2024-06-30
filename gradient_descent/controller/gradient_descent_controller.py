import os

from fastapi import APIRouter
from pydantic import BaseModel

import tensorflow as tf
import numpy as np

gradientDescentRouter = APIRouter()


class LinearRegressionModel(tf.Module):
    def __init__(self):
        self.W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
        self.b = tf.Variable(tf.zeros([1]))

    def __call__(self, x):
        return self.W * x + self.b


# 손실 함수 정의
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# 훈련 함수 정의
def train_model(model, X, y, learning_rate=0.01, num_epochs=1000):
    X_tensor = tf.constant(X, dtype=tf.float32)
    y_tensor = tf.constant(y, dtype=tf.float32)
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            y_pred = model(X_tensor)
            loss = mean_squared_error(y_tensor, y_pred)
        gradients = tape.gradient(loss, [model.W, model.b])
        model.W.assign_sub(gradients[0] * learning_rate)
        model.b.assign_sub(gradients[1] * learning_rate)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")
    return model


# 모델 저장 함수
def save_model(model, path):
    np.savez(path, W=model.W.numpy(), b=model.b.numpy())


# 모델 불러오기 함수
def load_model(path):
    model = LinearRegressionModel()
    data = np.load(path)
    model.W.assign(data['W'])
    model.b.assign(data['b'])
    return model


# 훈련 데이터 생성
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


# 훈련 엔드포인트
@gradientDescentRouter.post("/gradient-descent-train")
async def train():
    model = LinearRegressionModel()
    model = train_model(model, X, y)
    save_model(model, 'linear_regression_model.npz')
    return {"status": "Model trained and saved successfully"}


# 예측 요청 모델
class PredictRequest(BaseModel):
    X: list


# 예측 엔드포인트
@gradientDescentRouter.post("/gradient-descent-predict")
async def predict(request: PredictRequest):
    if not os.path.exists('linear_regression_model.npz'):
        return {"error": "Model not found. Train the model first."}

    model = load_model('linear_regression_model.npz')
    X_new = tf.constant(request.X, dtype=tf.float32)
    predictions = model(X_new).numpy().tolist()
    return {"predictions": predictions}
