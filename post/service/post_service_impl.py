from typing import List

from aiomysql import Pool

from post.controller.response_form.create_post_response_form import CreatePostResponseForm
from post.entity.models import Post
from post.repository.post_repository_impl import PostRepositoryImpl
from post.service.post_service import PostService
from post.service.request.create_post_request import CreatePostRequest
from post.service.response.create_post_response import CreatePostResponse


class PostServiceImpl(PostService):
    def __init__(self, db_pool: Pool):
        self.post_repository = PostRepositoryImpl(db_pool)

    async def create_post(self, create_post_request: CreatePostRequest) -> CreatePostResponseForm:
        post = create_post_request.toPost()
        post_id = await self.post_repository.create(post)
        createPostResponse = CreatePostResponse(id=post_id)
        return CreatePostResponseForm.fromCreatePostResponse(createPostResponse)

    async def list_posts(self) -> List[Post]:
        return await self.post_repository.list()
