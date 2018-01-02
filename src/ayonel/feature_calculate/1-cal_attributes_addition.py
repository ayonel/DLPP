'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
 新增一些属性，使用数据库更新
'''
# coding: utf-8
# 计算一些属性
from src.constants import *
from src.database.dbutil import *
from src.utils import *
import pymongo
import re
import sortedcontainers as sc


def build_empty_result_dict(pull_list):
    result_dict= {}
    for number in pull_list:
        number = str(number)
        result_dict[number] = {}
        # result_dict[number]['src_addition'] = 0     # ok
        # result_dict[number]['src_deletion'] = 0     # ok
        # result_dict[number]['text_code_proportion'] = 0     # ok
        # result_dict[number]['history_commit_review_time'] = 0     # ok
        # result_dict[number]['recent_3_month_pr'] = 0     # ok
        # result_dict[number]['recent_3_month_commit'] = 0     # ok
        # result_dict[number]['is_reviewer_commit'] = False
        # result_dict[number]['text_forward_link'] = False     # title和body中是否含有外链ok

        result_dict[number]['week'] = 0  # pr提交时是周几
        result_dict[number]['pr_file_merged_count'] = 0  # pr中的file的历史merge数
        result_dict[number]['pr_file_merged_proportion'] = 0  # pr中的file的历史merge率
        result_dict[number]['pr_file_rejected_count'] = 0  # pr中的file的历史reject数
        result_dict[number]['pr_file_rejected_proportion'] = 0  # pr中的file的历史reject率
        result_dict[number]['pr_file_submitted_count'] = 0  # pr中的file的历史pr提交数
        result_dict[number]['last_pr'] = True  # 上一个pr是接受还是拒绝
        result_dict[number]['last_10_pr_merged'] = 0  # 上10个pr中的merge数
        result_dict[number]['last_10_pr_rejected'] = 0  # 上10个pr中的拒绝数
    return result_dict


# 直接从pullinfo中能得到的四个属性
def cal_src(pull_list, pull_dict, result_dict):
    for pull in pull_list:
        result_dict[pull]['src_addition'] = pull_dict[pull]['additions']
        result_dict[pull]['src_deletion'] = pull_dict[pull]['deletions']
        result_dict[pull]['commits'] = pull_dict[pull]['commits']
    return result_dict

# 正则匹配： (9cc8c6f3730) 或者  (#26696)
def forward_link_match(text):
    return re.search(r"\([a-z0-9]{11}\)", text) or re.search(r"\(#\d+?\)", text)

# 计算body和title中是否包含外链
def text_forward_link(pull_list, pull_dict, result_dict):
    for pull in pull_list:
        result_dict[pull]['text_forward_link'] = True if forward_link_match(str(pull_dict[pull]['body'])+str(pull_dict[pull]['title'])) else False
    print("外链计算完毕")
    return result_dict


# 返回过滤掉代码之后的评论长度
def filter_code_len(text):
    string = re.sub(r'`.*?`', ' ', re.sub('[\r\n]', ' ', text))
    string = re.sub(r'`.*?`', ' ', re.sub('[\r\n]', ' ', string))
    return len(string)

# 返回代码比例
def text_code_proportion(pull_list, pull_dict, result_dict):
    for pull in pull_list:
        text = str(pull_dict[pull]['title']) + str(pull_dict[pull]['body'])
        result_dict[pull]['text_code_proportion'] = 1-round(filter_code_len(text)/len(text), 5)
    print("代码比例计算完毕")
    return result_dict

# 返回是否由内部人员提交
def is_reviewer_commit(pull_list, pull_dict, result_dict, reviewer_set):
    for pull in pull_list:
        if pull_dict[pull]['author'] in reviewer_set:
            result_dict[pull]['is_reviewer_commit'] = True
    return result_dict


# 历史评审时间
def review_time(pull_list, pull_dict, result_dict):
    committer_dict = {}

    for pull in pull_dict:
        committer = pull_dict[pull]['author']
        if committer not in committer_dict:
            committer_dict[committer] = set()
            committer_dict[committer].add(pull)
        else:
            committer_dict[committer].add(pull)

    for k, v in enumerate(pull_list):
        history_pass_commit_num = 0
        history_pass_commit_review_time = 0

        now_time = pull_dict[v]['created_at']
        committer = pull_dict[v]['author']
        this_number = str(pull_dict[v]['number'])
        for pull in committer_dict[committer]:
            if pull_dict[pull]['number'] < int(this_number):
                if pull_dict[pull]['merged'] and pull_dict[pull]['merged_at'] < now_time:
                    history_pass_commit_num += 1
                    history_pass_commit_review_time += (pull_dict[pull]['merged_at']-pull_dict[pull]['created_at'])/SECOND_1_HOUR + 1

        if history_pass_commit_num != 0:
            result_dict[this_number]['history_commit_review_time'] = round(history_pass_commit_review_time/history_pass_commit_num,5)
        else:
            result_dict[this_number]['history_commit_review_time'] = 100000
    print("review time计算完毕")
    return result_dict

# 计算项目最近的3个月的接受的pr数、commit数
def recent_event(client, pull_list, pull_dict, result_dict):

    # 构造一个list,格式为:(FLAG, created_at)
    # FLAG:0----pr,  1----commit
    L = []
    for pull in pull_dict:
        L.append((0, str(pull), pull_dict[pull]['created_at']))
    commit_list = client[org]['commit'].find({'commit_at': {'$lt': STMAP_2016_7_31}})
    for commit in commit_list:
        L.append((1, commit['sha'], commit['commit_at']))

    L.sort(key=lambda x: x[2])
    print(org+"的pr、commit总数为："+str(len(L)))

    # 保存L中pr的索引
    index_dict = {}
    for k, v in enumerate(L):
        if v[0] == 0:
            index_dict[v[1]] = k

    for pull in pull_list:
        recent_pr = 0
        recent_commit = 0
        index = index_dict[pull]
        lasttime = pull_dict[pull]['created_at'] - SECOND_3_MONTH
        # 从index位置倒序向前
        for i in range(index-1, -1, -1):
            if L[i][2] < lasttime:
                break
            if L[i][0] == 1:
                recent_commit += 1
            else:
                recent_pr += 1
        result_dict[pull]['recent_3_month_pr'] = recent_pr
        result_dict[pull]['recent_3_month_commit'] = recent_commit

    print("recent event计算完毕")
    return result_dict


# 计算该条pr提交的日期是星期几
def week(pullinfo_list):
    for pullinfo in pullinfo_list:
        result_dict[str(pullinfo['number'])]['week'] = getWeekFromStamp(pullinfo['created_at'])
    return result_dict


# 构造字典，键为文件名，值为列表[(关闭时间, 关闭类型（0，拒绝；1，接受）)，按时间排序]
def get_file_close_dict(pull_dict):
    file_close_dict = {}

    for pull in pull_dict:
        closed_at = pull_dict[pull]['closed_at']
        for file in pull_dict[pull]['prfile']:
            if file in file_close_dict:
                if pull_dict[pull]['merged']:
                    file_close_dict[file].add((closed_at, True))
                else:
                    file_close_dict[file].add((closed_at, False))
            else:
                file_close_dict[file] = sc.SortedList()
                if pull_dict[pull]['merged']:
                    file_close_dict[file].add((closed_at, True))
                else:
                    file_close_dict[file].add((closed_at, False))
    return file_close_dict


# 统计文件提交数，文件merge数，文件reject数，文件merge率，文件reject率
def file_stat(pull_dict, file_close_dict):
    for pull in pull_dict:
        created_at = pull_dict[pull]['created_at']
        pr_file_submitted_count = 0
        pr_file_merged_proportion = 0.0
        pr_file_merged_count = 0
        pr_file_rejected_proportion = 0.0
        pr_file_rejected_count = 0


        for file in pull_dict[pull]['prfile']:
            if 'test/' in file or 'spec/' in file or 'tests/' in file:  # file不是测试文件
                pass
            else:
                for history_file in file_close_dict[file]:
                    if history_file[0] < created_at:
                        if history_file[1]:  # merge
                            pr_file_merged_count += 1
                        else:
                            pr_file_rejected_count += 1
                        pr_file_submitted_count += 1

        if pr_file_submitted_count > 0:
            pr_file_merged_proportion = pr_file_merged_count / pr_file_submitted_count
            pr_file_rejected_proportion = pr_file_rejected_count / pr_file_submitted_count
        result_dict[pull]['pr_file_merged_count'] = pr_file_merged_count
        result_dict[pull]['pr_file_merged_proportion'] = pr_file_merged_proportion
        result_dict[pull]['pr_file_rejected_count'] = pr_file_rejected_count
        result_dict[pull]['pr_file_rejected_proportion'] = pr_file_rejected_proportion
        result_dict[pull]['pr_file_submitted_count'] = pr_file_submitted_count
    return result_dict


def last_pr(pull_dict, pull_list):
    for k, v in enumerate(pull_list):
        created_at = pull_dict[v]['created_at']
        last_10_pr_merged = 0
        last_10_pr_rejected = 0
        last_10_pr_count = 0
        last_pr = True if k == 0 else pull_dict[pull_list[k-1]]['merged']
        for index in range(k)[::-1]:
            if pull_dict[pull_list[index]]['closed_at'] < created_at:
                if last_10_pr_count == 10:
                    break
                if pull_dict[pull_list[index]]['merged']:
                    last_10_pr_merged += 1
                else:
                    last_10_pr_rejected += 1
                last_10_pr_count += 1

        result_dict[v]['last_10_pr_merged'] = last_10_pr_merged
        result_dict[v]['last_10_pr_rejected'] = last_10_pr_rejected
        result_dict[v]['last_pr'] = last_pr
    return result_dict


if __name__ == '__main__':
    client = get_connection()
    for org, repo in org_list:
        print(org)
        db = client[org]
        # 获取pull_list, pull_dict
        pullinfo_list = list(db['pullinfo'].find().sort('number', pymongo.ASCENDING))
        commit_list = list(db['commit'].find().sort('commit_at', pymongo.ASCENDING))
        pull_list = []
        pull_dict = {}
        # 为pull_dict添加prfile字段

        for pullinfo in pullinfo_list:
            pull_list.append(str(pullinfo['number']))
            pull_dict[str(pullinfo['number'])] = pullinfo

        for pull in pull_dict:
            pull_dict[pull]['prfile'] = []
        for pullfile in list(client[org]['pullfile'].find()):
            if pullfile['number'] in pull_dict:
                pull_dict[pullfile['number']]['prfile'].append(pullfile['filename'])

        # 获取file_merge_dict
        file_close_dict = get_file_close_dict(pull_dict)
        # reviewer_set = set([x['name'] for x in list(client[org]['reviewer'].find())])
        result_dict = build_empty_result_dict(pull_list)

        # 查询commitfile_list
        # commitfile_list = list(db['commitfile'].find())
        # 计算代码属性
        # result_dict = cal_src(pull_list, pull_dict, result_dict)
        # result_dict = text_forward_link(pull_list, pull_dict, result_dict)
        # result_dict = text_code_proportion(pull_list, pull_dict, result_dict)
        # result_dict = is_reviewer_commit(pull_list, pull_dict, result_dict, reviewer_set)
        # result_dict = review_time(pull_list, pull_dict, result_dict)
        # result_dict = recent_event(client, pull_list, pull_dict, result_dict)
        result_dict = week(pullinfo_list)
        result_dict = file_stat(pull_dict, file_close_dict)
        result_dict = last_pr(pull_dict, pull_list)
        for pr in result_dict:
            client[org]['ayonel'].update({'number': pr}, {'$set': result_dict[pr]})



