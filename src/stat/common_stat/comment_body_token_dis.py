# 计算每个项目的comment的token长度

from src.database.dbutil import *
from src.constants import *


@mongo
def comment_token_dis(client):
    for org, repo in org_list:
        print(org)
        pull_dict = {}

        if org in bot_reviewer:
            reviewer_set = set([x['name'] for x in list(client[org]['reviewer'].find())]) - set(bot_reviewer[org])
        else:
            reviewer_set = set([x['name'] for x in list(client[org]['reviewer'].find())])

        # pullcomment_list = list(client[org]['pullcomment'].find(
        #     {'inline': 0, 'created_at': {'$lt': STMAP_2016_7_31}, 'commentor': {'$in': list(reviewer_set)}})
        #                         .sort([('number', pymongo.ASCENDING), ('created_at', pymongo.ASCENDING)]))

        # for pullcomment in pullcomment_list:
        #     if int(pullcomment['number']) not in pull_dict:
        #         pull_dict[int(pullcomment['number'])] = pullcomment['body_token']
        #     else:
        #         pull_dict[int(pullcomment['number'])] += pullcomment['body_token']

        # 添加pull的body以及title信息

        pullinfo_list = list(client[org]['pullinfo'].find({'created_at': {'$lt': STMAP_2016_7_31}}))

        for pullinfo in pullinfo_list:
            if pullinfo['number'] not in pull_dict:
                pull_dict[pullinfo['number']] = pullinfo['body_token'] + pullinfo['title_token']
            else:
                pull_dict[pullinfo['number']] += pullinfo['body_token'] + pullinfo['title_token']

        # print(org+'的pull数为:%s' % str(len(pull_dict)))
        pull_token_size_list = []
        for pull in pull_dict:
            pull_token_size_list.append(str(pull)+':'+str(pull_dict[pull].count(' ')))
        pull_token_size_list.sort(key=lambda x: int(x.split(':')[1]))
        # print(pull_token_size_list[1].split(':')[1])
        length = len(pull_token_size_list)

        token_dis_dict = {}

        for i in range(1,20):
            seq_proportion = i*0.05
            token_dis_dict[str(int(seq_proportion*100))] = pull_token_size_list[int(length*seq_proportion)].split(':')[1]

        client[org]['stat'].update({},{'$set':{'pullinfo_word_distribution':token_dis_dict}})
        continue
        less250=0
        b5001k=0
        b250500=0
        b1k2k=0
        b2k3k=0
        b3k4k=0
        more4k=0


        for item in pull_token_size_list:
            if int(item.split(':')[1])>=4000:
                more4k+=1
            elif int(item.split(':')[1])>=3000 and int(item.split(':')[1]) < 4000:
                b3k4k += 1
            elif int(item.split(':')[1])>=2000 and int(item.split(':')[1]) < 3000:
                b2k3k += 1
            elif int(item.split(':')[1])>=1000 and int(item.split(':')[1]) < 2000:
                b1k2k += 1
            elif int(item.split(':')[1]) >= 500 and int(item.split(':')[1]) < 1000:
                b5001k += 1
            elif int(item.split(':')[1])>=250 and int(item.split(':')[1]) < 500:
                b250500 += 1
            else:
                less250 += 1
        print(org+'---  0-250:'+str(less250)+'  250-500:'+str(b250500)+'  500-1k:'+str(b5001k)+'  1k-2k:'+str(b1k2k)+'  2k-3k:'+str(b2k3k)+'  3k-4k:'+str(b3k4k)+'  4k~:'+str(more4k))



if __name__ == '__main__':
    comment_token_dis()


