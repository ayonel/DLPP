'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
 统计每个项目PR数以及每天产生的pr数
'''

from src.database.dbutil import *
from src.constants import *
import csv


@mongo
def pr(client):
    outfile = open("data/pr_num.csv", "w", newline="")
    writer = csv.writer(outfile)
    writer.writerow(["repo", "pr数", "日均pr数"])
    for org, repo in org_list:
        pullinfo_list = list(client[org]['pullinfo'].find({'created_at': {'$lt': STMAP_2016_7_31}}).sort('number', 1))
        pr_num = len(pullinfo_list)

        last_30 = (pullinfo_list[-1]['created_at']) - 30*24*3600
        count = 0
        for pr in pullinfo_list[::-1]:
            if(pr['created_at'] > last_30):
                count += 1
            else:
                break
        pr_num_ave = round(count/30, 3)
        writer.writerow([repo, pr_num, pr_num_ave])


@mongo
def check(client):
    pullinfo_list = list(client['rails']['pullinfo'].find())
    origin_set = set()
    for pullinfo in pullinfo_list:
        origin_set.add(pullinfo['number'])

    ayonel_list = list(client['rails']['ayonel'].find())
    ayonel_set = set()
    for pullinfo in ayonel_list:
        ayonel_set.add(int(pullinfo['number']))

    gousios_list = list(client['rails']['gousios'].find())
    gousios_set = set()
    for pullinfo in gousios_list:
        gousios_set.add(int(pullinfo['number']))

    print('原始pr数：'+str(len(origin_set)))
    print('ayonel的pr数：'+str(len(ayonel_set)))
    print('gousios的pr数：'+str(len(gousios_set)))
    print(len(origin_set&ayonel_set))
    print(len(origin_set&gousios_set))
    print(len(ayonel_set&gousios_set))


if __name__ == '__main__':
    check()
