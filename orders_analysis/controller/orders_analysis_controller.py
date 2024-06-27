import joblib
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

import tensorflow as tf
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

from orders_analysis.service.orders_analysis_service_impl import OrdersAnalysisServiceImpl

ordersAnalysisRouter = APIRouter()

async def injectOrdersAnalysisService() -> OrdersAnalysisServiceImpl:
    return OrdersAnalysisServiceImpl()

@ordersAnalysisRouter.get("/orders-train")
async def orders_data_train(ordersAnalysisService: OrdersAnalysisServiceImpl =
                               Depends(injectOrdersAnalysisService)):

    try:
        result = await ordersAnalysisService.train_model()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))



class ViewCount(BaseModel):
    count: float


@ordersAnalysisRouter.post("/orders-predict")
def orders_data_predict(request: ViewCount):
    try:
        scaler = joblib.load("scaler.pkl")
        print("is it here ?")

        predictions = []
        for i in range(1, 11):
            model = tf.keras.models.load_model(f"model_{i}.h5")
            X_pred = np.array([[request.count]])
            X_pred_scaled = scaler.transform(X_pred)
            prediction = model.predict(X_pred_scaled).flatten()[0]
            predictions.append(float(prediction))

        avg_prediction = np.mean(predictions)

        return JSONResponse(content={"predicted_quantity": avg_prediction})

    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model file not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

