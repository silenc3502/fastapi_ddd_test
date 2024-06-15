from fastapi import APIRouter, Depends
from typing import List
from aiomysql import Pool

from post.controller.response_form.create_post_response_form import CreatePostResponseForm
from post.service.post_service_impl import PostServiceImpl
from post.service.request.create_post_request import CreatePostRequest
from post.entity.models import Post
from async_db.database import getMysqlPool  # Assuming this function is defined to get db pool
from post.service.response.create_post_response import CreatePostResponse

post_router = APIRouter()

# Dependency to get PostService instance
async def get_post_service(db_pool: Pool = Depends(getMysqlPool)) -> PostServiceImpl:
    return PostServiceImpl(db_pool)

@post_router.post("/", response_model=CreatePostResponseForm)
async def create_post(post_request: CreatePostRequest, post_service: PostServiceImpl = Depends(get_post_service)):
    createPostResponseForm = await post_service.create_post(post_request)
    return createPostResponseForm

@post_router.get("/", response_model=List[Post])
async def read_posts(post_service: PostServiceImpl = Depends(get_post_service)):
    posts = await post_service.list_posts()
    return posts
