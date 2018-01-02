import src.database.CONFIG as CONFIG
import pymongo
import functools


# 获取数据库连接
def get_connection():
    return pymongo.MongoClient(host=CONFIG.MONGODB_HOST, port=CONFIG.MONGODB_PORT)


# 关闭数据库连接
def close_connection(client):
    client.close()


# 更改某个字段名
def rename_field(client, db_name, col_name, old_field_name, new_filed_name):
    client[db_name][col_name].update({}, {'$rename': {old_field_name: new_filed_name}}, False, True)


# 注解，数据库连接装饰器
def mongo(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        client = get_connection()
        result = func(client, *args, **kw)
        close_connection(client)
        return result
    return wrapper

