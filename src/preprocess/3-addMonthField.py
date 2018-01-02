# 为pullinfo表添加month字段，标志该pr创建时属于哪一月，从每个项目的第一个PR创建的时间算起

from src.constants import *
from src.database.dbutil import *
from src.utils import getTimestamp


@mongo
def main(client):
    for org, repo in org_list:
        first_pr_created_at = getTimestamp(first_pr_time[org])

        pullinfo_list = list(client[org]['pullinfo'].find())
        for pullinfo in pullinfo_list:
            month = int((pullinfo['created_at'] - first_pr_created_at) / SECOND_1_MONTH) + 1
            client[org]['pullinfo'].update({'_id': pullinfo['_id']}, {'$set': {'month': month}})

if __name__ == '__main__':
    main()
