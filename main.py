# from fastapi import FastAPI
# from post.controller import post_controller
# from initializer.domain_initializer import initEachDomain
#
# app = FastAPI()
#
# app.include_router(post_controller.controller, prefix="/posts")
#
# @app.on_event("startup")
# async def startupEvent():
#     await initEachDomain(app)
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=33333)
import os

import nltk
from dotenv import load_dotenv
# from fastapi import FastAPI, Depends, HTTPException
# from pydantic import BaseModel
# from aiomysql import create_pool, Pool
# from typing import List
#
# app = FastAPI()
#
# class Post(BaseModel):
#     id: int = None
#     title: str
#     content: str
#
# async def get_db_pool():
#     if not hasattr(app.state, "db_pool"):
#         app.state.db_pool = await create_pool(
#             host='localhost',
#             port=3306,
#             user='eddi',
#             password='eddi@123',
#             db='fastapi_test_db',
#             minsize=1,
#             maxsize=10
#         )
#     return app.state.db_pool
#
# @app.on_event("shutdown")
# async def shutdown_event():
#     app.state.db_pool.close()
#     await app.state.db_pool.wait_closed()
#
# @app.post("/posts/", response_model=Post)
# async def create_post(post: Post, db_pool: Pool = Depends(get_db_pool)):
#     async with db_pool.acquire() as conn:
#         async with conn.cursor() as cur:
#             await cur.execute(
#                 "INSERT INTO posts (title, content) VALUES (%s, %s)",
#                 (post.title, post.content)
#             )
#             await conn.commit()
#             await cur.execute("SELECT LAST_INSERT_ID()")
#             post_id = await cur.fetchone()
#             post.id = post_id[0]
#             return post
#
# @app.get("/posts/", response_model=List[Post])
# async def read_posts(db_pool: Pool = Depends(get_db_pool)):
#     async with db_pool.acquire() as conn:
#         async with conn.cursor() as cur:
#             await cur.execute("SELECT id, title, content FROM posts")
#             result = await cur.fetchall()
#             posts = [Post(id=row[0], title=row[1], content=row[2]) for row in result]
#             return posts
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=33333)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from exponential_regression.controller.exponential_regression_controller import exponentialRegressionRouter
from kmeans_analysis.controller.kmeans_controller import kmeansRouter
from logistic_regression.controller.logistic_regression_controller import logisticRegressionRouter
from natural_language_processing.controller.natural_language_processing_controller import \
    naturalLanguageProcessingRouter
from polynomial_regression.controller.polynomial_regression_controller import polynomialRegressionRouter
from post.controller.post_controller import post_router
from async_db.database import getMysqlPool
from random_forest.controller.random_forest_controller import randomForestRouter
from random_number.controller.random_number_controller import randomNumberRouter
from train_test_evaluation.controller.train_test_evaluation_controller import trainTestEvaluationRouter
from word_cloud.controller.word_cloud_controller import wordCloudRouter
from async_db.database import createTableIfNecessary


async def lifespan(app: FastAPI):
    # Startup event
    app.state.db_pool = await getMysqlPool()
    await createTableIfNecessary(app.state.db_pool)

    yield

    # Shutdown event
    app.state.db_pool.close()
    await app.state.db_pool.wait_closed()


app = FastAPI(lifespan=lifespan)

# 데이터베이스 연결 설정
# @app.on_event("startup")
# async def startup_event():
#     app.state.db_pool = await getMysqlPool()
#
# @app.on_event("shutdown")
# async def shutdown_event():
#     app.state.db_pool.close()
#     await app.state.db_pool.wait_closed()

load_dotenv()

# CORS settings
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

# origins = [
#     "http://localhost:8081",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NLTK stopwords 다운로드 확인 및 필요 시 다운로드
def download_nltk_data():
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    if not os.path.exists(os.path.join(nltk_data_path, "corpora", "stopwords")):
        nltk.download('stopwords', download_dir=nltk_data_path)

download_nltk_data()

# 라우터 등록
app.include_router(post_router, prefix="/posts")
app.include_router(randomNumberRouter, prefix="/random-number")
app.include_router(logisticRegressionRouter)
app.include_router(trainTestEvaluationRouter)
app.include_router(polynomialRegressionRouter)
app.include_router(exponentialRegressionRouter)
app.include_router(randomForestRouter)
app.include_router(wordCloudRouter)
app.include_router(naturalLanguageProcessingRouter)
app.include_router(kmeansRouter)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=33333)
