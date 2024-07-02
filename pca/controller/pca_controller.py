from typing import Dict, Any

import joblib
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

pcaRouter = APIRouter()


@pcaRouter.get("/pca", response_model=Dict[str, Any])
async def perform_pca(n_points: int = 200, n_features: int = 5, n_components: int = 2):
    np.random.seed(42)
    mean = np.zeros(n_features)
    cov = np.random.rand(n_features, n_features)
    cov = np.dot(cov, cov.transpose())

    data = np.random.multivariate_normal(mean, cov, n_points)
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(1, n_features + 1)])

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)

    original_data = df.values.tolist()
    pca_data = principal_components.tolist()
    explained_variance_ratio = pca.explained_variance_ratio_.tolist()

    return {
        "original_data": original_data,
        "pca_data": pca_data,
        "explained_variance_ratio": explained_variance_ratio,
    }