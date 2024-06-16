from fastapi import APIRouter
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

polynomialRegressionRouter = APIRouter()


# 임의의 데이터 생성 함수
def generate_data():
    np.random.seed(0)
    X = 2 - 3 * np.random.normal(0, 1, 100)
    y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 100)
    X = X[:, np.newaxis]
    return X, y


# 다항 회귀 수행 함수
def perform_polynomial_regression():
    X, y = generate_data()

    # 다항 피처 생성 (2차 항 포함)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # 선형 회귀 모델에 다항 피처 적합
    model = LinearRegression()
    model.fit(X_poly, y)

    # 모델 예측
    X_new = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_new_poly = poly.transform(X_new)
    y_pred = model.predict(X_new_poly)

    return X.flatten().tolist(), y.tolist(), X_new.flatten().tolist(), y_pred.tolist()


@polynomialRegressionRouter.get("/polynomial-regression")
def perform_regression():
    X, y, X_new, y_pred = perform_polynomial_regression()
    return {
        "X": X,
        "y": y,
        "X_new": X_new,
        "y_pred": y_pred
    }