# coding: utf-8
# 计算一些属性

from src.constants import *
from src.database.dbutil import *
import pymongo
import re




def build_empty_result_dict(pull_list):
    result_dict= {}
    for number in pull_list:
        number = str(number)
        result_dict[number] = {}
        result_dict[number]['history_commit_num'] = 0    # ok
        result_dict[number]['src_churn'] = 0     # ok
        result_dict[number]['history_commit_passrate'] = 0.0    # ok
        result_dict[number]['recent_project_passrate'] = 0.0     # ok
        result_dict[number]['num_particpants'] = 0     # ok
        result_dict[number]['files_changes'] = 0    # ok
        result_dict[number]['inline_comment_num'] = 0   # ok
        result_dict[number]['is_reviewer_commit'] = False
        result_dict[number]['has_test'] = False     # ok
        result_dict[number]['forward_link'] = False     # ok
        result_dict[number]['has_body'] = False    # ok
        result_dict[number]['is_approved'] = False      # TODO
    return result_dict


# 直接从prinfo中能得到的四个属性
def direct_info(pull_list, pull_dict, result_dict):
    for pull in pull_list:
        result_dict[pull]['has_body'] = (pull_dict[pull]['body'] != '')
        result_dict[pull]['src_churn'] = pull_dict[pull]['additions'] + pull_dict[pull]['deletions']
        result_dict[pull]['files_changes'] = pull_dict[pull]['changed_files']
        result_dict[pull]['inline_comment_num'] = pull_dict[pull]['review_comments']
    return result_dict


# 是否包含测试
def cal_has_test(client, org, pull_list, result_dict):
    # 需要查数据库, 先取出所有，构造成字典
    pullfile_connection = client[org]['pullfile']
    pullfile_list = pullfile_connection.find()
    pullfile_dict = {}
    for pullfile in pullfile_list:
        if pullfile['number'] not in pullfile_dict:
            pullfile_dict[pullfile['number']] = []
            pullfile_dict[pullfile['number']].append(pullfile)
        else:
            pullfile_dict[pullfile['number']].append(pullfile)
    for pull in pull_list:
        this_has_test = False
        if pull not in pullfile_dict:  # 没有文件的pull
            continue
        for file in pullfile_dict[pull]:
            if 'test/' in file['filename'] or 'spec/' in file['filename'] or 'tests/' in file['filename']:
                this_has_test = True
                break
        result_dict[pull]['has_test'] = this_has_test

    return result_dict


# pr评论参与人数， 是否包含外链
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
        this_forward_link = False
        if pull not in pullcomment_dict:
            result_dict[pull]['num_particpants'] = 0
            result_dict[pull]['forward_link'] = False
            continue

        this_num_particpants = len(set([x['commentor'] for x in pullcomment_dict[pull]]))
        body_list = [x['body'] for x in pullcomment_dict[pull]]
        for item in body_list:
            if item:
                if re.search('#\d+', item):
                    this_forward_link = True
                    break
        result_dict[pull]['num_particpants'] = this_num_particpants
        result_dict[pull]['forward_link'] = this_forward_link
    return result_dict


def cal_passrate(pull_list, pull_dict, result_dict):
    committer_dict = {}

    for pull in pull_dict:
        committer = pull_dict[pull]['author']
        if committer not in committer_dict:
            committer_dict[committer] = set()
            committer_dict[committer].add(pull)
        else:
            committer_dict[committer].add(pull)


    for k, v in enumerate(pull_list):
        history_commit_num = 0
        history_pass_commit_num = 0
        recent_project_pull_num = 0
        recent_project_passed_pull_num = 0
        now_time = pull_dict[v]['created_at']
        committer = pull_dict[v]['author']
        this_number = str(pull_dict[v]['number'])
        last_3_month_time = pull_dict[v]['created_at'] - SECOND_3_MONTH
        # 计算history_commit_num以及history_commit_passrate
        for pull in committer_dict[committer]:
            if pull_dict[pull]['number'] < int(this_number):
                history_commit_num += 1
                if pull_dict[pull]['merged'] and pull_dict[pull]['merged_at'] < now_time:
                    history_pass_commit_num += 1

        # 计算history_project_passrate
        if k == 0:
            pass
        else:
            for history_pull in pull_list[k-1::-1]:
                if pull_dict[history_pull]['created_at'] > last_3_month_time:
                    if pull_dict[history_pull]['closed_at'] < now_time:
                        recent_project_pull_num += 1
                        if pull_dict[history_pull]['merged']:
                            recent_project_passed_pull_num += 1
                else:
                    break

        if recent_project_pull_num != 0:
            result_dict[this_number]['recent_project_passrate'] = recent_project_passed_pull_num / recent_project_pull_num
        else:
            result_dict[this_number]['recent_project_passrate'] = 0.0

        if history_commit_num != 0:
            result_dict[this_number]['history_commit_passrate'] = history_pass_commit_num / history_commit_num
        else:
            result_dict[this_number]['history_commit_passrate'] = 0.0

        result_dict[this_number]['history_commit_num'] = history_commit_num

    return result_dict


# 主函数
if __name__ == '__main__':
    client = get_connection()
    for org, repo in org_list:
        print(org)
        db = client[org]
        # 获取pull_list, pull_dict
        pullinfo_list = list(db['pullinfo'].find().sort('number', pymongo.ASCENDING))
        pull_list = []
        pull_dict = {}
        for pullinfo in pullinfo_list:
            pull_list.append(str(pullinfo['number']))
            pull_dict[str(pullinfo['number'])] = pullinfo
        result_dict = build_empty_result_dict(pull_list)

        result_dict = cal_passrate(pull_list, pull_dict, result_dict)

        # 计算四个直接属性 has_body src_churn files_changes inline_comment_num
        result_dict = direct_info(pull_list, pull_dict, result_dict)
        # 计算has_test
        result_dict = cal_has_test(client, org, pull_list, result_dict)
        print('test_churn 计算完毕')
        # 计算参与人数， 是否包含外链
        result_dict = attr_in_comments(client, org, pull_list, result_dict)
        print('num_comments num_particpants conflict forward_link计算完毕')
        # 计算提交者历史通过率，历史提交次数，项目最近通过率
        result_dict = cal_passrate(pull_list,pull_dict,result_dict)
        print(result_dict)
        # 开始保存
        for result in result_dict:
            result_dict[result]['number'] = result
            client[org]['ayonel'].insert(result_dict[result])
