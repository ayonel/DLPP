# coding: utf-8
# 计算gousios论文中的pullrequest属性

import pymongo
import src.database.dbutil as dbutil
from src.constants import *
import re


# 直接从prinfo中能得到的三个属性
def direct_three(pull_list, pull_dict, result_dict):
    for pull in pull_list:
        result_dict[pull]['num_commits'] = pull_dict[pull]['commits']
        result_dict[pull]['src_churn'] = pull_dict[pull]['additions'] + pull_dict[pull]['deletions']
        result_dict[pull]['files_changes'] = pull_dict[pull]['changed_files']
    return result_dict


# pr中的测试代码(改动)行数
def test_churn(client, org, pull_list, result_dict):
    # 需要查数据库, 先取出所有，构造成字典
    pullfile_list = client[org]['pullfile'].find()
    pullfile_dict = {}
    for pullfile in pullfile_list:
        if pullfile['number'] not in pullfile_dict:
            pullfile_dict[pullfile['number']] = []
            pullfile_dict[pullfile['number']].append(pullfile)
        else:
            pullfile_dict[pullfile['number']].append(pullfile)
    for pull in pull_list:
        # if pull not in pullfile_dict:
        #     result_dict.pop(pull)
        #     continue
        this_test_churn = 0
        if pull in pullfile_dict:
            for file in pullfile_dict[pull]:
                if 'test/' in file['filename'] or 'spec/' in file['filename'] or 'tests/' in file['filename']:
                    this_test_churn += file['changes']
            result_dict[pull]['test_churn'] = this_test_churn
    return result_dict


# pr中的评论总数，评论参与人数， 是否包含conflict，是否包含外链
def attr_in_comments(client, org, pull_list, result_dict):
    # 需要查数据库
    pullcomment_connection = client[org]['pullcomment']
    pullcomment_list = pullcomment_connection.find()
    pullcomment_dict = {}
    for pullcomment in pullcomment_list:
        if pullcomment['number'] not in pullcomment_dict:
            pullcomment_dict[pullcomment['number']] = []
            pullcomment_dict[pullcomment['number']].append(pullcomment)
        else:
            pullcomment_dict[pullcomment['number']].append(pullcomment)

    for pull in pull_list:
        if pull not in result_dict:
            continue
        # if pull not in pullcomment_dict:
        #     result_dict.pop(pull)
        #     continue
        if pull in pullcomment_dict:
            this_confilt = False
            this_forward_link = False
            this_num_comments = len(pullcomment_dict[pull])
            this_num_particpants = len(set([x['commentor'] for x in pullcomment_dict[pull]]))
            body_list = [x['body'] for x in pullcomment_dict[pull]]
            for item in body_list:
                if item:
                    if 'confilt' in item:
                        this_confilt = True
                        break
            for item in body_list:
                if item:
                    if re.search('#\d+', item):
                        this_forward_link = True
                        break
            result_dict[pull]['num_comments'] = this_num_comments
            result_dict[pull]['num_particpants'] = this_num_particpants
            result_dict[pull]['conflict'] = this_confilt
            result_dict[pull]['forward_link'] = this_forward_link
    return result_dict



def build_empty_result_dict(pull_list):
    result_dict= {}
    for number in pull_list:
        number = str(number)
        result_dict[number] = {}
        result_dict[number]['num_commits'] = 0
        result_dict[number]['src_churn'] = 0
        result_dict[number]['test_churn'] = 0
        result_dict[number]['files_changes'] = 0
        result_dict[number]['num_comments'] = 0
        result_dict[number]['num_particpants'] = 0
        result_dict[number]['conflict'] = False
        result_dict[number]['forward_link'] = False
    return result_dict


if __name__ == '__main__':
    client = dbutil.get_connection()
    for org, repo in org_list:
        print(org)
        # 先建立数据库连接
        db = client[org]
        # 获取pull_list, pull_dict
        pullinfo_list = list(db['pullinfo'].find().sort('number', pymongo.ASCENDING))
        pull_list = []
        pull_dict = {}
        for pullinfo in pullinfo_list:
            pull_list.append(str(pullinfo['number']))
            pull_dict[str(pullinfo['number'])] = pullinfo
        result_dict = build_empty_result_dict(pull_list)
        # 计算三个直接属性 num_commits src_churn files_changes
        result_dict = direct_three(pull_list, pull_dict, result_dict)
        print('num_commits src_churn files_changes 计算完毕')
        # 计算test_churn
        # 8846--》 8807
        result_dict = test_churn(client, org, pull_list, result_dict)
        print('test_churn 计算完毕')
        # 计算num_comments
        result_dict = attr_in_comments(client, org, pull_list, result_dict)
        print('num_comments num_particpants conflict forward_link计算完毕')
        # 开始保存
        for result in result_dict:
            result_dict[result]['number'] = result
            client[org]['gousios'].insert(result_dict[result])
