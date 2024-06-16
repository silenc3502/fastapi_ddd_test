import aiomysql

from dotenv import load_dotenv
import os

load_dotenv()


async def getMysqlPool():
    return await aiomysql.create_pool(
        host=os.getenv('MYSQL_HOST'),
        port=int(os.getenv('MYSQL_PORT')),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        db=os.getenv('MYSQL_DB'),
        autocommit=True
    )
