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
        result_dict[number] = {}
        result_dict[number]['history_pr_num_decay'] = 0  # ok
        result_dict[number]['history_pass_pr_num'] = 0  # ok
        result_dict[number]['history_pass_pr_num_decay'] = 0  # ok
        result_dict[number]['recent_1_month_project_pr_num'] = 0  # ok
        result_dict[number]['recent_1_month_project_pr_num_decay'] = 0  # ok
        result_dict[number]['recent_3_month_project_pr_num'] = 0     # ok
        result_dict[number]['recent_3_month_project_pr_num_decay'] = 0

        result_dict[number]['pr_file_merged_count_decay'] = 0  # pr中的file的历史merge数
        result_dict[number]['pr_file_rejected_count_decay'] = 0  # pr中的file的历史reject数
        result_dict[number]['pr_file_submitted_count_decay'] = 0  # pr中的file的历史pr提交数
        result_dict[number]['last_10_pr_merged_decay'] = 0  # 上10个pr中的merge数
        result_dict[number]['last_10_pr_rejected_decay'] = 0  # 上10个pr中的拒绝数
    return result_dict


def cal_history_pr_num_decay(pull_list, pull_dict, result_dict):
    committer_dict = {}

    for pull in pull_dict:
        committer = pull_dict[pull]['author']
        if committer not in committer_dict:
            committer_dict[committer] = set()
            committer_dict[committer].add(pull)
        else:
            committer_dict[committer].add(pull)

    for k, v in enumerate(pull_list):
        history_pr_num_decay = 0
        history_pass_pr_num_decay = 0
        history_pass_pr_num = 0
        recent_1_month_project_pr_num = 0
        recent_1_month_project_pr_num_decay = 0
        recent_3_month_project_pr_num = 0
        recent_3_month_project_pr_num_decay = 0

        now_time = pull_dict[v]['created_at']
        committer = pull_dict[v]['author']
        this_number = str(pull_dict[v]['number'])
        last_3_month_time = pull_dict[v]['created_at'] - SECOND_3_MONTH
        last_1_month_time = pull_dict[v]['created_at'] - SECOND_1_MONTH
        # 计算history_commit_num以及history_commit_passrate

        for pull in committer_dict[committer]:
            if pull_dict[pull]['number'] < int(this_number):
                history_pr_num_decay += 1/((now_time-pull_dict[pull]['created_at']) / SECOND_1_DAY + 1)
                if pull_dict[pull]['merged'] and pull_dict[pull]['merged_at'] < now_time:
                    history_pass_pr_num += 1
                    history_pass_pr_num_decay += 1/(int((now_time-pull_dict[pull]['closed_at']) / SECOND_1_DAY) + 1)

        if k == 0:
            pass
        else:
            for history_pull in pull_list[k-1::-1]:
                if pull_dict[history_pull]['created_at'] > last_3_month_time:
                    if pull_dict[history_pull]['closed_at'] < now_time:
                        recent_3_month_project_pr_num += 1
                        recent_3_month_project_pr_num_decay += 1/(int((now_time-pull_dict[history_pull]['created_at']) / SECOND_1_DAY) + 1)
                        if pull_dict[history_pull]['merged']:
                            recent_3_month_project_pr_num_decay += 1
                else:
                    break

        if k == 0:
            pass
        else:
            for history_pull in pull_list[k-1::-1]:
                if pull_dict[history_pull]['created_at'] > last_1_month_time:
                    if pull_dict[history_pull]['closed_at'] < now_time:
                        recent_1_month_project_pr_num += 1
                        recent_1_month_project_pr_num_decay += 1/(int((now_time-pull_dict[history_pull]['created_at']) / SECOND_1_DAY) + 1)
                        if pull_dict[history_pull]['merged']:
                            recent_1_month_project_pr_num_decay += 1
                else:
                    break

        result_dict[this_number]['recent_1_month_project_pr_num'] = recent_1_month_project_pr_num
        result_dict[this_number]['recent_1_month_project_pr_num_decay'] = recent_1_month_project_pr_num_decay
        result_dict[this_number]['recent_3_month_project_pr_num'] = recent_3_month_project_pr_num
        result_dict[this_number]['recent_3_month_project_pr_num_decay'] = recent_3_month_project_pr_num_decay
        result_dict[this_number]['history_pr_num_decay'] = history_pr_num_decay
        result_dict[this_number]['history_pass_pr_num'] = history_pass_pr_num
        result_dict[this_number]['history_pass_pr_num_decay'] = history_pass_pr_num_decay
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
def file_stat_decay(pull_dict, file_close_dict):
    for pull in pull_dict:
        created_at = pull_dict[pull]['created_at']
        pr_file_submitted_count = 0
        pr_file_submitted_count_decay = 0
        pr_file_merged_count_decay = 0
        pr_file_rejected_count_decay = 0

        for file in pull_dict[pull]['prfile']:
            if 'test/' in file or 'spec/' in file or 'tests/' in file:  # file不是测试文件
                pass
            else:
                for history_file in file_close_dict[file]:
                    if history_file[0] < created_at:
                        if history_file[1]:  # merge
                            pr_file_merged_count_decay += 1/(int((created_at-history_file[0])/SECOND_1_DAY) + 1)
                        else:
                            pr_file_rejected_count_decay += 1/(int((created_at-history_file[0])/SECOND_1_DAY) + 1)
                        pr_file_submitted_count_decay += 1/(int((created_at-history_file[0])/SECOND_1_DAY) + 1)
                        pr_file_submitted_count += 1

        result_dict[pull]['pr_file_merged_count_decay'] = pr_file_merged_count_decay
        result_dict[pull]['pr_file_rejected_count_decay'] = pr_file_rejected_count_decay
        result_dict[pull]['pr_file_submitted_count_decay'] = pr_file_submitted_count_decay
    return result_dict


def last_pr_decay(pull_dict, pull_list):
    for k, v in enumerate(pull_list):
        created_at = pull_dict[v]['created_at']
        last_10_pr_merged_decay = 0
        last_10_pr_rejected_decay = 0
        last_10_pr_count = 0
        for index in range(k)[::-1]:
            closed_at = pull_dict[pull_list[index]]['closed_at']
            if closed_at < created_at:
                if last_10_pr_count == 10:
                    break
                if pull_dict[pull_list[index]]['merged']:
                    last_10_pr_merged_decay += 1/(int((created_at-closed_at)/SECOND_1_DAY)+1)
                else:
                    last_10_pr_rejected_decay += 1/(int((created_at-closed_at)/SECOND_1_DAY)+1)

        result_dict[v]['last_10_pr_merged_decay'] = last_10_pr_merged_decay
        result_dict[v]['last_10_pr_rejected_decay'] = last_10_pr_rejected_decay
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


        result_dict = cal_history_pr_num_decay(pull_list, pull_dict, result_dict)
        result_dict = file_stat_decay(pull_dict, file_close_dict)
        result_dict = last_pr_decay(pull_dict, pull_list)
        # 查询commitfile_list
        commitfile_list = list(db['commitfile'].find())
        # 计算代码属性
        for pr in result_dict:
            client[org]['ayonel'].update({'number': pr}, {'$set': result_dict[pr]})



