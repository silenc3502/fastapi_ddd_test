from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

import numpy as np
from sklearn.linear_model import LinearRegression

exponentialRegressionRouter = APIRouter()

@exponentialRegressionRouter.get("/exponential-regression")
def perform_exponential_regression():
    try:
        # 임의의 데이터 생성
        np.random.seed(0)
        X = np.arange(1, 100)
        y = 2 * np.exp(0.1 * X) + np.random.normal(size=X.size)

        # 지수 회귀 모델 적합 (로그 변환)
        log_y = np.log(y)
        model = LinearRegression()
        model.fit(X[:, np.newaxis], log_y)

        # 모델 예측
        X_new = np.linspace(1, 100, 100)
        y_pred = np.exp(model.predict(X_new[:, np.newaxis]))

        # 결과 반환 (NumPy 데이터 유형을 Python 기본 유형으로 변환)
        result = {
            "original_data": list(map(lambda x: [int(x[0]), float(x[1])], zip(X, y))),
            "predicted_data": list(map(lambda x: [float(x[0]), float(x[1])], zip(X_new, y_pred))),
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_)
        }
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
