'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
'''

from src.database.dbutil import *
from src.constants import *
import csv

@mongo
def commentor_num_dis(client):
    outfile = open('data/commentor_dis.csv', "w", newline="")
    writer = csv.writer(outfile)
    writer.writerow(['org', '0', '1', '2', '3', '4', '5', '>5'])
    for org, repo in org_list:
        print(org)
        pullinfo_list = list(
            client[org]['pullinfo'].find({'created_at': {'$lt': STMAP_2016_7_31}}).sort('number', pymongo.ASCENDING))
        pull_dict = {}
        pullcomment_dict = {}
        for pullinfo in pullinfo_list:
            pull_dict[pullinfo['number']] = pullinfo
            pullcomment_dict[pullinfo['number']] = set()

        pullcomment_list = list(client[org]['pullcomment'].find())
        for pullcomment in pullcomment_list:
            number = int(pullcomment['number'])
            author = pull_dict[number]['author']
            if pullcomment['commentor'] != author:
                pullcomment_dict[number].add(pullcomment['commentor'])

        zero = 0
        one = 0
        two = 0
        three = 0
        four = 0
        five = 0
        more_than_five = 0

        for pull in pullcomment_dict:
            commentor_num = len(pullcomment_dict[pull])
            if commentor_num == 0:
                zero+=1
            elif commentor_num == 1:
                one+=1
            elif commentor_num == 2:
                two+=1
            elif commentor_num == 3:
                three+=1
            elif commentor_num == 4:
                four+=1
            elif commentor_num == 5:
                five+=1
            else:
                more_than_five+=1

        zero = round(zero / len(pullcomment_dict), 5)
        one = round(one / len(pullcomment_dict), 5)
        two = round(two / len(pullcomment_dict), 5)
        three = round(three / len(pullcomment_dict), 5)
        four = round(four / len(pullcomment_dict), 5)
        five = round(five / len(pullcomment_dict), 5)
        more_than_five = round(more_than_five / len(pullcomment_dict), 5)
        writer.writerow([org, zero, one, two, three, four, five, more_than_five])


@mongo
def reviewer_not_comment_dis(client):
    outfile = open('data/reviewer_not_comment.csv', "w", newline="")
    writer = csv.writer(outfile)
    writer.writerow(['org', 'reviewer_not_comment'])
    for org, repo in org_list:
        print(org)
        pullinfo_list = list(
            client[org]['pullinfo'].find({'created_at': {'$lt': STMAP_2016_7_31}}).sort('number', pymongo.ASCENDING))
        pull_dict = {}
        pullcomment_dict = {}
        for pullinfo in pullinfo_list:
            pull_dict[pullinfo['number']] = pullinfo
            pullcomment_dict[pullinfo['number']] = set()

        pullcomment_list = list(client[org]['pullcomment'].find())
        for pullcomment in pullcomment_list:
            number = int(pullcomment['number'])
            author = pull_dict[number]['author']
            if pullcomment['commentor'] != author:
                pullcomment_dict[number].add(pullcomment['commentor'])

        reviewer_set = set(x['name'] for x in list(client[org]['reviewer'].find()))
        print(org+':'+str(len(reviewer_set)))
        reviewer_comment_num = 0
        for pull in pullcomment_dict:
            if len(pullcomment_dict[pull] & reviewer_set) == 0:
                reviewer_comment_num += 1
        writer.writerow([org, round(reviewer_comment_num/len(pullcomment_dict), 5)])


if __name__ == '__main__':
    # commentor_num_dis()
    reviewer_not_comment_dis()




