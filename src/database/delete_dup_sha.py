# 为commit表删除重复sha
import src.database.dbutil as dbutil
import src.constants as constants

if __name__ == '__main__':
    client = dbutil.get_connection()
    for org, repo in constants.org_list:
        result = list(client[org]['commit'].find())
        D = {}
        S = set()
        for item in result:
            if item['sha'] not in D:
                D[item['sha']] = 1
            else:
                S.add(item['sha'])
        for s in S:
            client[org]['commit'].delete_one({'sha':s, 'additions':{'$exists':False}})

