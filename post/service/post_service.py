from abc import ABC, abstractmethod
from typing import List, Optional

from post.controller.response_form.create_post_response_form import CreatePostResponseForm
from post.entity.models import Post
from post.service.request.create_post_request import CreatePostRequest


class PostService(ABC):
    @abstractmethod
    async def create_post(self, create_post_request: CreatePostRequest) -> CreatePostResponseForm:
        pass

    @abstractmethod
    async def list_posts(self) -> List[Post]:
        pass

    @abstractmethod
    async def get_post_by_id(self, post_id: int) -> Optional[Post]:
        pass
