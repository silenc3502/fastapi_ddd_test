from fastapi import APIRouter, Depends, Request
from aiomysql import Pool
from typing import List

from post.entity.models import Post

post_router = APIRouter()

async def get_db_pool(request: Request) -> Pool:
    return request.app.state.db_pool

@post_router.post("/", response_model=Post)
async def create_post(post: Post, db_pool: Pool = Depends(get_db_pool)):
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO posts (title, content) VALUES (%s, %s)",
                (post.title, post.content)
            )
            await conn.commit()
            await cur.execute("SELECT LAST_INSERT_ID()")
            post_id = await cur.fetchone()
            return {**post.dict(), "id": post_id[0]}

@post_router.get("/", response_model=List[Post])
async def read_posts(db_pool: Pool = Depends(get_db_pool)):
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, title, content FROM posts")
            result = await cur.fetchall()
            posts = [{"id": row[0], "title": row[1], "content": row[2]} for row in result]
            return posts
