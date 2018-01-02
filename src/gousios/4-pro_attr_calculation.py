# coding: utf-8
# 计算gousios论文中的project属性

import pymongo
import src.database.dbutil as dbutil
import src.constants as constants


# 在pull request创建时的项目中的总代码行数
# 计算方法：从 第一条commit一直计算到pull 创建时的前一条commit，累加additions和deletions，以及pr创建前仓库中的测试代码行数*1000/总代码行数
def sloc_AND_test_lines_per_kloc(pullinfo_list, commit_list, result_dict):
    # 构造一个sloc字典
    sloc_dict = {}
    for c_k, c_v in enumerate(commit_list):
        sloc_dict[c_k] = {}
        if c_k == 0:
            sloc_dict[c_k]['sloc'] = c_v['additions'] - c_v['deletions']
            sloc_dict[c_k]['test_lines_per_kloc'] = c_v['test_additions'] - c_v['test_deletions']
        else:
            # 需要判断，有些commit没有爬取到commitfile信息，所以其additions以及deletions没有， 可以补齐数据库
            sloc_dict[c_k]['sloc'] = sloc_dict[c_k-1]['sloc'] + c_v['additions'] - c_v['deletions']
            sloc_dict[c_k]['test_lines_per_kloc'] = sloc_dict[c_k-1]['test_lines_per_kloc']+c_v['test_additions'] - c_v['test_deletions']
    for p_k, p_v in enumerate(pullinfo_list):
        last_commit_index = p_v['last_commit_index']
        if p_v['last_commit_index'] == -1:
            result_dict[p_v['number']]['sloc'] = 0
            result_dict[p_v['number']]['test_lines_per_kloc'] = 0
        else:
            result_dict[p_v['number']]['sloc'] = sloc_dict[last_commit_index]['sloc']
            if result_dict[p_v['number']]['sloc'] == 0:
                result_dict[p_v['number']]['test_lines_per_kloc'] = 0
            else:
                result_dict[p_v['number']]['test_lines_per_kloc'] = (sloc_dict[last_commit_index]['test_lines_per_kloc'] / result_dict[p_v['number']]['sloc']) * 1000
    return result_dict


# pull创建前的前三个月内直接提交commit的人，需要排除pull中的commit
def team_size_AND_perc_ext_contribs(pullinfo_list, commit_list, pull_link_commit_dict, result_dict):
    for p_k, p_v in enumerate(pullinfo_list):
        # 获取前三个月从pull中来的commit
        pullcommit_set = set()
        pullcommit_author_set = set()
        if p_v['first_pull_index'] == -1:
            pass
        else:
            for i in range(p_v['first_pull_index'], p_k):
                for commit in pullinfo_list[i]['commit']:
                    pullcommit_set.add(commit['sha'])
                    pullcommit_author_set.add(commit['author'])

        # 获取前三个月正常的commit
        commit_set = set()
        commit_author_set = set()
        if pull_link_commit_dict[p_k][0] == -1:
            pass
        else:
            for i in range(pull_link_commit_dict[p_k][0], pull_link_commit_dict[p_k][1]+1):
                commit_set.add(commit_list[i]['sha'])
                commit_author_set.add(commit_list[i]['author'])
        team_size = len(commit_author_set-pullcommit_author_set)
        if len(commit_set) == 0:
            perc_ext_contribs = 0.0
        else:
            perc_ext_contribs = (len(commit_set & pullcommit_set) / len(commit_set))*100

        result_dict[p_v['number']]['team_size'] = team_size
        result_dict[p_v['number']]['perc_ext_contribs'] = perc_ext_contribs
    return result_dict


# pr创建前的三个月内在该pr的file上的commit数量
def commits_files_touched(pullinfo_list, commit_list, pull_link_commit_dict, result_dict):
    for p_k, p_v in enumerate(pullinfo_list):
        commit_num = 0
        this_file_set = p_v['fileinfo']
        if pull_link_commit_dict[p_k][0] == -1:
            pass
        else:
            for i in range(pull_link_commit_dict[p_k][0], pull_link_commit_dict[p_k][1]+1):
                if len(commit_list[i]['fileinfo'] & this_file_set) != 0:
                    commit_num += 1
        result_dict[p_v['number']]['commits_files_touched'] = commit_num
    return result_dict


# 构造空结果字典
def build_empty_result_dict(pullinfo_list):
    result_dict = {}
    for pullinfo in pullinfo_list:
        number = pullinfo['number']
        result_dict[number] = {}
        result_dict[number]['sloc'] = 0
        result_dict[number]['team_size'] = 0
        result_dict[number]['perc_ext_contribs'] = 0
        result_dict[number]['commits_files_touched'] = 0
        result_dict[number]['test_lines_per_kloc'] = 0
    return result_dict

# 为pullinfo_list添加其前三个月最早的一条pr的index 以及commit信息
def add_first_pull_index(pullinfo_list, pullcommit_dict):
    first_pull_index = 0
    for p_k, p_v in enumerate(pullinfo_list):
        time_prior_3_month = p_v['created_at'] - constants.SECOND_3_MONTH
        while (pullinfo_list[first_pull_index]['created_at'] < time_prior_3_month):
            first_pull_index += 1
        pullinfo_list[p_k]['first_pull_index'] = first_pull_index - 1
        if p_v['number'] not in pullcommit_dict:
            pullinfo_list[p_k]['commit'] = []
        else:
            pullinfo_list[p_k]['commit'] = pullcommit_dict[p_v['number']]
    return pullinfo_list

# 构造pullcommit_dict
def construct_pullcommit_dict(pullcommit_list):
    pullcommit_dict = {}
    for pullcommit in pullcommit_list:
        if int(pullcommit['number']) not in pullcommit_dict:
            pullcommit_dict[int(pullcommit['number'])] = []
            pullcommit_dict[int(pullcommit['number'])].append(pullcommit)
        else:
            pullcommit_dict[int(pullcommit['number'])].append(pullcommit)
    return pullcommit_dict

# 构造commitfile_dict
def construct_commitfile_dict(commitfile_list):
    commitfile_dict = {}
    for commitfile in commitfile_list:
        if commitfile['commit_sha'] not in commitfile_dict:
            commitfile_dict[commitfile['commit_sha']] = []
            commitfile_dict[commitfile['commit_sha']].append(commitfile)
        else:
            commitfile_dict[commitfile['commit_sha']].append(commitfile)
    return commitfile_dict

# 为commit_list 添加test_additions以及test_deletions属性以及fileinfo
def add_test_attr_AND_fileinfo(commit_list, commitfile_dict):
    for c_k, c_v in enumerate(commit_list):
        commit_list[c_k]['fileinfo'] = set()
        test_additions = 0
        test_deletions = 0
        if c_v['sha'] in commitfile_dict:
            for file in commitfile_dict[c_v['sha']]:
                commit_list[c_k]['fileinfo'].add(file['filename'])
                if 'test/' in file['filename'] or 'spec/' in file['filename'] or 'tests/' in file['filename']:
                    test_additions += file['additions']
                    test_deletions += file['deletions']

        commit_list[c_k]['test_additions'] = test_additions
        commit_list[c_k]['test_deletions'] = test_deletions


    return commit_list

# 构造pullfile_dict
def construct_pullfile_dict(pullfile_list):
    pullfile_dict = {}
    for pullfile in pullfile_list:
        if int(pullfile['number']) not in pullfile_dict:
            pullfile_dict[int(pullfile['number'])] = set()
            pullfile_dict[int(pullfile['number'])].add(pullfile['filename'])
        else:
            pullfile_dict[int(pullfile['number'])].add(pullfile['filename'])
    return pullfile_dict


# 添加last_commit_index属性用于计算sloc以及test_lines_per_kloc
def add_last_commit_index(pullinfo_list, commit_list):
    commit_index = 0
    commit_list_length = len(commit_list)
    for p_k, p_v in enumerate(pullinfo_list):
        while (commit_index < commit_list_length and p_v['created_at'] > commit_list[commit_index]['commit_at']):
            commit_index += 1
        pullinfo_list[p_k]['last_commit_index'] = commit_index - 1
    return pullinfo_list


# 构造一个以pull_index 为键，该pull前三个月的第一个commit以及最后一个commit的index为值的字典
def construct_pull_link_commit_dict(pullinfo_list, commit_list):
    commit_start_index = 0
    commit_end_index = 0
    commit_list_length = len(commit_list)
    pull_link_commit_dict = {}
    for p_k, p_v in enumerate(pullinfo_list):
        pull_link_commit_dict[p_k] = []
        pull_create_time = p_v['created_at']
        time_prior_3_month = p_v['created_at'] - constants.SECOND_3_MONTH
        while (commit_start_index < commit_list_length and time_prior_3_month > commit_list[commit_start_index][
            'commit_at']):
            commit_start_index += 1
        while (commit_end_index < commit_list_length and pull_create_time > commit_list[commit_end_index]['commit_at']):
            commit_end_index += 1
        pull_link_commit_dict[p_k].append(commit_start_index)
        pull_link_commit_dict[p_k].append(commit_end_index - 1)
    return pull_link_commit_dict

# 为pullinfo_list 添加fileinfo字段
def add_fileinfo(pullinfo_list, pullfile_dict):
    for p_k, p_v in enumerate(pullinfo_list):
        if p_v['number'] not in pullfile_dict:
            pullinfo_list[p_k]['fileinfo'] = set()
        else:
            pullinfo_list[p_k]['fileinfo'] = pullfile_dict[p_v['number']]
    return pullinfo_list



if __name__ == '__main__':
    client = dbutil.get_connection()
    for org, repo in constants.org_list:
        print(org)
        # 先建立数据库连接
        db = client[org]
        #  pull_dict
        pullinfo_list = list(db['pullinfo'].find().sort('number', pymongo.ASCENDING))

        # 获取pull中的commit
        pullcommit_list = list(db['pullcommit'].find())
        pullcommit_dict = construct_pullcommit_dict(pullcommit_list)

        # 为pullinfo_list添加其前三个月最早的一条pr的index,添加first_pull_index属性,以及commit信息
        pullinfo_list = add_first_pull_index(pullinfo_list, pullcommit_dict)

        # 为pullinfo_list 添加fileinfo字段
        pullfile_list = list(db['pullfile'].find())
        pullfile_dict = construct_pullfile_dict(pullfile_list)
        pullinfo_list = add_fileinfo(pullinfo_list, pullfile_dict)

        # 构造commit_list
        commit_list = list(db['commit'].find().sort('commit_at', pymongo.ASCENDING))

        # 查询commitfile_list
        commitfile_list = list(db['commitfile'].find())
        commitfile_dict = construct_commitfile_dict(commitfile_list)

        # 为commit_list 添加test_additions以及test_deletions属性 以及fileinfo属性
        commit_list = add_test_attr_AND_fileinfo(commit_list, commitfile_dict)

        # 添加last_commit_index属性用于计算sloc以及test_lines_per_kloc
        pullinfo_list = add_last_commit_index(pullinfo_list, commit_list)

        # 构造一个以pull_index 为键，该pull前三个月的第一个commit以及最后一个commit的index为值的字典
        pull_link_commit_dict = construct_pull_link_commit_dict(pullinfo_list, commit_list)

        # 构造空结果字典
        result_dict = build_empty_result_dict(pullinfo_list)
        # 计算 sloc以及test_lines_per_kloc
        result_dict = sloc_AND_test_lines_per_kloc(pullinfo_list, commit_list, result_dict)
        print('sloc以及test_lines_per_kloc 计算完毕')

        # 计算team_size 以及 perc_ext_contribs
        result_dict = team_size_AND_perc_ext_contribs(pullinfo_list, commit_list, pull_link_commit_dict, result_dict)
        print('team_size以及_perc_ext_contribs计算完毕')

        # 计算commits_files_touched
        result_dict = commits_files_touched(pullinfo_list, commit_list, pull_link_commit_dict, result_dict)
        print('commits_files_touched计算完毕')

        # 开始输出
        for result in result_dict:
            client[org]['gousios'].update({'number': str(result)}, {'$set': result_dict[result]})








