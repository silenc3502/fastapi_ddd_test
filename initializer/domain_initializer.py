from fastapi import FastAPI
from post.repository.post_repository_impl import PostRepositoryImpl
from post.service.post_service_impl import PostServiceImpl
from async_db.database import getMysqlPool

async def initEachDomain(app: FastAPI):
    mysqlPool = await getMysqlPool()

    post_repository = PostRepositoryImpl(mysqlPool)
    post_service = PostServiceImpl(post_repository)

    app.state.post_repository = post_repository
    app.state.post_service = post_service
