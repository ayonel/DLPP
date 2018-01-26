# 统计每个项目的PR拒绝比例
from src.database import dbutil
from src.constants import org_list

if __name__ == '__main__':
    client = dbutil.get_connection()
    L = []
    for org, repo in org_list:
        merged = client[org]['pullinfo'].find({'merged': True}).count()
        whole = client[org]['pullinfo'].find({}).count()
        L.append(repo+','+str(1-merged/whole))
    L.sort(key=lambda x: float(x.split(',')[1]), reverse=True)
    for item in L:
        print(item)
    dbutil.close_connection(client)


