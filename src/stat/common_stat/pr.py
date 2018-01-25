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


@mongo
def comment(client):
    comment_num_list = []
    for org, repo in org_list:
        comment_num_list.append(client[org]['pullcomment'].find().count())

    print(sum(comment_num_list))
    print(sorted(comment_num_list))

import requests
import json
def star():
    star_num_list = []
    for org, repo in org_list:
        star_num = json.loads(requests.get('https://api.github.com/repos/'+org+'/'+repo).text)['stargazers_count']
        print(org+','+str(star_num))
        star_num_list.append(star_num)
    print(sum(star_num_list))
    print(sorted(star_num_list))

from collections import Counter
def language():
    lan_list = []
    for org, repo in org_list:
        lan_dict = json.loads(requests.get('https://api.github.com/repos/'+org+'/'+repo+'/languages?access_token=a2c8a20da495eb45b56710a459f12f1a5b34577e').text)
        max = 0
        for lan in lan_dict:
            if int(lan_dict[lan]) > max:
                max = int(lan_dict[lan])

        print(str(lan_dict)+','+org+'/'+repo)
        for lan in lan_dict:
            if lan_dict[lan] == max:
                lan_list.append(lan)
                print(org+','+lan)
                break
    counter = Counter(lan_list)
    print(counter)

@mongo
def count(client):
    outfile = open("data/count.csv", "w", newline="")
    writer = csv.writer(outfile)
    writer.writerow(["repo", "pr数", "pr评论数", "pr文件数", "pr commit数", "commit","commitfile,reviewer"])
    for org, repo in org_list:
        pr_num = client[org]['pullinfo'].find().count()
        pr_comment_num = client[org]['pullcomment'].find().count()
        pr_file_num = client[org]['pullfile'].find().count()
        pr_commit_num = client[org]['pullcommit'].find().count()
        commit_num = client[org]['commit'].find().count()
        commit_file_num = client[org]['commitfile'].find().count()
        reviewer_num = client[org]['reviewer'].find().count()
        writer.writerow([repo, pr_num,pr_comment_num,pr_file_num,pr_commit_num,commit_num,commit_file_num,reviewer_num])

if __name__ == '__main__':
    count()