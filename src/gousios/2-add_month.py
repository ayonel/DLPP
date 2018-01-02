# 为每个pull添加其月份属性,分别添加到pullinfo表以及gousios表

import pymongo
import src.database.dbutil as dbutil
import src.constants as constants


if __name__ == '__main__':
    client = dbutil.get_connection()
    for org, repo in constants.org_list:
        print(org)
        pullinfo_list = list(client[org]['pullinfo'].find())
        monthDict = {}
        for pullinfo in pullinfo_list:
            monthDict[pullinfo['number']] = int((pullinfo['created_at'] - constants.first_issue_time_stamp[org]) / (30*24*3600)) + 1

        for pull in monthDict:
            client[org]['gousios'].update({'number':str(pull)}, {'$set': {'month': monthDict[pull]}})



