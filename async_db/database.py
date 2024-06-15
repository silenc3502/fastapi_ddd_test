import aiomysql

MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_USER = 'eddi'
MYSQL_PASSWORD = 'eddi@123'
MYSQL_DB = 'fastapi_test_db'

async def getMysqlPool():
    return await aiomysql.create_pool(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DB,
        autocommit=True
    )
