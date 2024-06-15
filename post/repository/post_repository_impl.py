from typing import List, Optional

from aiomysql import Pool
from post.repository.post_repository import PostRepository
from post.entity.models import Post

class PostRepositoryImpl(PostRepository):
    def __init__(self, db_pool: Pool):
        self.db_pool = db_pool

    async def create(self, post: Post) -> int:
        async with self.db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO posts (title, content) VALUES (%s, %s)",
                    (post.title, post.content)
                )
                await conn.commit()
                await cur.execute("SELECT LAST_INSERT_ID()")
                post_id = await cur.fetchone()
                return post_id[0]

    async def list(self) -> List[Post]:
        async with self.db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT id, title, content FROM posts")
                result = await cur.fetchall()
                posts = [Post(id=row[0], title=row[1], content=row[2]) for row in result]
                return posts

    async def findById(self, post_id: int) -> Optional[Post]:
        async with self.db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT id, title, content FROM posts WHERE id = %s", (post_id,))
                result = await cur.fetchone()
                if result:
                    return Post(id=result[0], title=result[1], content=result[2])
                else:
                    return None
