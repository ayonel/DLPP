# 为commit表补齐additions以及deletions字段
# 不要运行，运行了就会增加重复数据，还得运行delete_dup_sha.py
import src.database.dbutil as dbutil
import src.constants as constants

if __name__ == '__main__':
    client = dbutil.get_connection()
    for org, repo in constants.org_list:
        result = list(client[org]['commit'].find({'additions': {'$exists': False}}))
        for item in result:
            client[org]['commit'].update({'sha': item['sha']}, {'$set': {'additions':0, 'deletions':0}})