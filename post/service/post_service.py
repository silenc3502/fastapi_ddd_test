from abc import ABC, abstractmethod
from typing import List

from post.controller.response_form.create_post_response_form import CreatePostResponseForm
from post.entity.models import Post
from post.service.request.create_post_request import CreatePostRequest
from post.service.response.create_post_response import CreatePostResponse


class PostService(ABC):
    @abstractmethod
    async def create_post(self, create_post_request: CreatePostRequest) -> CreatePostResponseForm:
        pass

    @abstractmethod
    async def list_posts(self) -> List[Post]:
        pass
