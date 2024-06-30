import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import tensorflow as tf
# pip install tensorflow_decision_forests
# pip install --upgrade typing_extensions
import tensorflow_decision_forests as tfdf

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from tensorflow_decision_forests.keras import RandomForestModel

decisionForestRouter = APIRouter()


# 입력 데이터 스키마 정의
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

# 모델 저장 경로
MODEL_PATH = "wine_model.h5"
FEATURE_NAMES_PATH = "wine_feature_names.joblib"
TARGET_NAMES_PATH = "wine_target_names.joblib"


@decisionForestRouter.get('/wine-info-excel')
def makeWineInfoExcel():
    wine = load_wine()

    # Create a DataFrame from the dataset
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target

    # Save DataFrame to Excel
    excel_file = "wine_dataset.xlsx"
    df.to_excel(excel_file, index=False)

    print(f"Wine dataset saved to {excel_file}")

    return {"message": "wine 정보 엑셀로 만들기"}

# 모델 학습 함수
@decisionForestRouter.post("/wine-train")
def train():
    # 데이터 준비
    wine = load_wine()
    data = pd.DataFrame(wine.data, columns=wine.feature_names)
    data['target'] = wine.target

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_df[wine.feature_names] = scaler.fit_transform(train_df[wine.feature_names])
    test_df[wine.feature_names] = scaler.transform(test_df[wine.feature_names])

    train_tf = tf.data.Dataset.from_tensor_slices(
        (dict(train_df.drop("target", axis=1)), train_df["target"].astype(int)))
    test_tf = tf.data.Dataset.from_tensor_slices((dict(test_df.drop("target", axis=1)), test_df["target"].astype(int)))
    train_tf = train_tf.batch(32)
    test_tf = test_tf.batch(32)

    # 모델 학습
    model = tfdf.keras.RandomForestModel(num_trees=100, max_depth=12, min_examples=6)
    model.fit(train_tf)

    # 모델 저장
    dump(model, MODEL_PATH)

    # 특성 이름 저장
    dump(wine.feature_names, FEATURE_NAMES_PATH)

    # 타겟 이름 저장
    dump(wine.target_names, TARGET_NAMES_PATH)

    return {"message": "Model trained and saved successfully."}


# 모델 로드 함수
def load_model():
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    else:
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")

# 특성 이름 로드 함수
def load_feature_names():
    if os.path.exists(FEATURE_NAMES_PATH):
        return load(FEATURE_NAMES_PATH)
    else:
        raise HTTPException(status_code=404, detail="Feature names not found. Please train the model first.")


# 타겟 이름 로드 함수
def load_target_names():
    if os.path.exists(TARGET_NAMES_PATH):
        return load(TARGET_NAMES_PATH)
    else:
        raise HTTPException(status_code=404, detail="Target names not found. Please train the model first.")


# 예측 엔드포인트 정의
@decisionForestRouter.post("/wine-predict")
async def predict(features: WineFeatures):
    # model = load_model()
    model = load_model()
    feature_names = load_feature_names()
    target_names = load_target_names()

    input_data = pd.DataFrame([[features.alcohol, features.malic_acid, features.ash, features.alcalinity_of_ash,
                                features.magnesium, features.total_phenols, features.flavanoids,
                                features.nonflavanoid_phenols, features.proanthocyanins,
                                features.color_intensity, features.hue, features.od280_od315_of_diluted_wines,
                                features.proline]], columns=feature_names)

    predictions = model.predict(input_data)
    predicted_class = int(predictions[0].argmax())
    predicted_class_name = target_names[predicted_class]

    return {"predicted_class": predicted_class, "predicted_class_name": predicted_class_name}