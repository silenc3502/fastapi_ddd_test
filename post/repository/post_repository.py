from abc import ABC, abstractmethod
from typing import List, Optional

from post.entity.models import Post
from post.service.request.create_post_request import CreatePostRequest
from post.service.response.create_post_response import CreatePostResponse


class PostRepository(ABC):
    @abstractmethod
    async def create(self, post: Post) -> int:
        pass

    @abstractmethod
    async def list(self) -> List[Post]:
        pass

    @abstractmethod
    async def findById(self, post_id: int) -> Optional[Post]:
        pass
