from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

from kmeans_analysis.controller.response_form.cluster_response_form import ClusterResponseForm

kmeansRouter = APIRouter()

@kmeansRouter.get("/kmeans-test", response_model=ClusterResponseForm)
async def kmeans_test():
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=4, n_init=10)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.tolist()
    labels = kmeans.labels_.tolist()
    points = X.tolist()
    print(f"points: {points}")
    return {"centers": centers, "labels": labels, "points": points}
