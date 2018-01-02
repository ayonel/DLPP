'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
'''

from src.database.dbutil import *
from src.constants import *
from src.utils import *
import nltk
import re

bug_set = set(bug_word_list)
Stemmer = nltk.PorterStemmer()

@mongo
def remove_self_close(client):
    writer = getCSVWriter("data/reject_no_self.csv", "w")
    writer.writerow(['org','repo', '原PR数', '筛选PR数','原拒绝率', '筛选拒绝率'])
    for org, repo in org_list:
        pipeline = [
                {
                    "$project": {
                        "author":1,
                        "merged_by":1,
                        "merged":1,
                        "diff":{"$ne":["$author", "$merged_by"]},
                        "time_limit":{"$lt":["$created_at", STMAP_2016_7_31]}
                    }
                },
                {
                    "$match":{
                        # "$and" : [{"diff": True, "time_limit":True}]
                        "diff": True,
                        "time_limit": True,
                    }
                }
        ]
        # 查询出提交者和关闭者非同一个人的PR
        reject_num = 0
        pullinfo_list = list(client[org]['pullinfo'].aggregate(pipeline))
        for pullinfo in pullinfo_list:
            if not pullinfo['merged']:
                reject_num += 1
        whole_num = client[org]['pullinfo'].find().count()
        writer.writerow([org,repo, whole_num, len(pullinfo_list), '', round(reject_num/len(pullinfo_list), 6)])

@mongo
def some_number(client):
    '''
    统计接受与不接受的pr之间一些技术变量的差异
    title长度
    body长度
    commit数
    文件数
    代码改动数
    :param client: 
    :return: 
    '''
    writer = getCSVWriter("data/some_number.csv", "w")
    writer.writerow(['org', 'repo',
                     '接受title长度', '拒绝title长度', '',
                     '接受body长度', '拒绝body长度', '',
                     '接受text长度', '拒绝text长度', '',
                     '接受commit数', '拒绝commit数', '',
                     '接受文件数', '拒绝文件数', '',
                     '接受代码add', '拒绝代码add', '',
                     '接受代码delete', '拒绝代码delete', '',
                     '接受代码churn', '拒绝代码churn', '',
                     ])
    for org, repo in org_list:
        pullinfo_list = list(client[org]['pullinfo'].find())
        merged_count = 0
        rejected_count = 0
        merged_body_len = 0
        rejected_body_len = 0
        merged_title_len = 0
        rejected_title_len = 0
        merged_text_len = 0
        rejected_text_len = 0
        merged_commit = 0
        rejected_commit = 0
        merged_filenum = 0
        rejected_filenum = 0
        merged_src_addition = 0
        rejected_src_addition = 0
        merged_src_deletion = 0
        rejected_src_deletion = 0
        merged_src_churn = 0
        rejected_src_churn = 0


        for pullinfo in pullinfo_list:
            if pullinfo['merged']:
                merged_count += 1
                merged_title_len += 0 if pullinfo['title_token'] == '' else pullinfo['title_token'].count(' ')
                merged_body_len += 0 if pullinfo['body_token'] == '' else pullinfo['body_token'].count(' ')
                merged_text_len += (0 if pullinfo['body_token'] == '' else pullinfo['body_token'].count(' ')) + (0 if pullinfo['title_token'] == '' else pullinfo['title_token'].count(' '))
                merged_commit += pullinfo['commits']
                merged_filenum += pullinfo['changed_files']
                merged_src_addition += pullinfo['additions']
                merged_src_deletion += pullinfo['deletions']
                merged_src_churn += pullinfo['additions'] + pullinfo['deletions']
            else:
                rejected_count += 1
                rejected_title_len += 0 if pullinfo['title_token'] == '' else pullinfo['title_token'].count(' ')
                rejected_body_len += 0 if pullinfo['body_token'] == '' else pullinfo['body_token'].count(' ')
                rejected_text_len += (0 if pullinfo['body_token'] == '' else pullinfo['body_token'].count(' ')) + (0 if pullinfo['title_token'] == '' else pullinfo['title_token'].count(' '))
                rejected_commit += pullinfo['commits']
                rejected_filenum += pullinfo['changed_files']
                rejected_src_addition += pullinfo['additions']
                rejected_src_deletion += pullinfo['deletions']
                rejected_src_churn += pullinfo['additions'] + pullinfo['deletions']
        writer.writerow([
            org,
            repo,
            round(merged_title_len/merged_count,5),
            round(rejected_title_len/rejected_count,5),
            '',
            round(merged_body_len / merged_count, 5),
            round(rejected_body_len / rejected_count, 5),
            '',
            round(merged_text_len / merged_count, 5),
            round(rejected_text_len / rejected_count, 5),
            '',
            round(merged_commit / merged_count, 5),
            round(rejected_commit / rejected_count, 5),
            '',
            round(merged_filenum / merged_count, 5),
            round(rejected_filenum / rejected_count, 5),
            '',
            round(merged_src_addition / merged_count, 5),
            round(rejected_src_addition / rejected_count, 5),
            '',
            round(merged_src_deletion / merged_count, 5),
            round(rejected_src_deletion / rejected_count, 5),
            '',
            round(merged_src_churn / merged_count, 5),
            round(rejected_src_churn / rejected_count, 5),
            ''
        ])

def get_medium(L):
    L = list(sorted(L))
    if len(L) % 2 == 1:
        return L[int(len(L)/2)]
    else:
        return round((L[int(len(L)/2-1)] + L[int(len(L)/2)]) / 2, 5)


@mongo
def some_number_medium(client):
    '''
    统计接受与不接受的pr之间一些技术变量的差异
    title长度
    body长度
    commit数
    文件数
    代码改动数
    :param client: 
    :return: 
    '''
    writer = getCSVWriter("data/some_number_medium.csv", "w")
    writer.writerow(['org', 'repo',
                     '接受title长度', '拒绝title长度', '',
                     '接受body长度', '拒绝body长度', '',
                     '接受text长度', '拒绝text长度', '',
                     '接受commit数', '拒绝commit数', '',
                     '接受文件数', '拒绝文件数', '',
                     '接受代码add', '拒绝代码add', '',
                     '接受代码delete', '拒绝代码delete', '',
                     '接受代码churn', '拒绝代码churn', '',
                     ])
    for org, repo in org_list:
        pullinfo_list = list(client[org]['pullinfo'].find())
        merged_count = 0
        rejected_count = 0
        merged_body_len = []
        rejected_body_len = []
        merged_title_len = []
        rejected_title_len = []
        merged_text_len = []
        rejected_text_len = []
        merged_commit = []
        rejected_commit = []
        merged_filenum = []
        rejected_filenum = []
        merged_src_addition = []
        rejected_src_addition = []
        merged_src_deletion = []
        rejected_src_deletion = []
        merged_src_churn = []
        rejected_src_churn = []


        for pullinfo in pullinfo_list:
            if pullinfo['merged']:
                merged_count += 1
                merged_title_len.append(0 if pullinfo['title_token'] == '' else pullinfo['title_token'].count(' '))
                merged_body_len.append(0 if pullinfo['body_token'] == '' else pullinfo['body_token'].count(' '))
                merged_text_len.append((0 if pullinfo['body_token'] == '' else pullinfo['body_token'].count(' ')) + (0 if pullinfo['title_token'] == '' else pullinfo['title_token'].count(' ')))
                merged_commit.append(pullinfo['commits'])
                merged_filenum.append(pullinfo['changed_files'])
                merged_src_addition.append(pullinfo['additions'])
                merged_src_deletion.append(pullinfo['deletions'])
                merged_src_churn.append(pullinfo['additions'] + pullinfo['deletions'])
            else:
                rejected_count += 1
                rejected_title_len.append(0 if pullinfo['title_token'] == '' else pullinfo['title_token'].count(' '))
                rejected_body_len.append(0 if pullinfo['body_token'] == '' else pullinfo['body_token'].count(' '))
                rejected_text_len.append((0 if pullinfo['body_token'] == '' else pullinfo['body_token'].count(' ')) + (0 if pullinfo['title_token'] == '' else pullinfo['title_token'].count(' ')))
                rejected_commit.append(pullinfo['commits'])
                rejected_filenum .append(pullinfo['changed_files'])
                rejected_src_addition.append(pullinfo['additions'])
                rejected_src_deletion.append(pullinfo['deletions'])
                rejected_src_churn.append(pullinfo['additions'] + pullinfo['deletions'])

        merged_title_len_medium = get_medium(merged_body_len)
        rejected_title_len_medium = get_medium(rejected_body_len)
        merged_body_len_medium = get_medium(merged_title_len)
        rejected_body_len_medium = get_medium(rejected_title_len)
        merged_text_len_medium = get_medium(merged_text_len)
        rejected_text_len_medium = get_medium(rejected_text_len)
        merged_commit_medium = get_medium(merged_commit)
        rejected_commit_medium = get_medium(rejected_commit)
        merged_filenum_medium = get_medium(merged_filenum)
        rejected_filenum_medium = get_medium(rejected_filenum)
        merged_src_addition_medium = get_medium(merged_src_addition)
        rejected_src_addition_medium = get_medium(rejected_src_addition)
        merged_src_deletion_medium = get_medium(merged_src_deletion)
        rejected_src_deletion_medium = get_medium(rejected_src_deletion)
        merged_src_churn_medium = get_medium(merged_src_churn)
        rejected_src_churn_medium = get_medium(rejected_src_churn)




        writer.writerow([
            org,
            repo,
            merged_body_len_medium,
            rejected_body_len_medium,
            '',
            merged_title_len_medium,
            rejected_title_len_medium,
            '',
            merged_text_len_medium,
            rejected_text_len_medium,
            '',
            merged_commit_medium,
            rejected_commit_medium,
            '',
            merged_filenum_medium,
            rejected_filenum_medium,
            '',
            merged_src_addition_medium,
            rejected_src_addition_medium,
            '',
            merged_src_deletion_medium,
            rejected_src_deletion_medium,
            '',
            merged_src_churn_medium,
            rejected_src_churn_medium,
            ''
        ])

@mongo
def some_boolean(client):
    '''
    统计在body长度为0的pr中，拒绝与接受的统计占比
    是否是内部人员
    是否含有bug词汇
    是否包含测试文件
    :param client: 
    :return: 
    '''
    writer = getCSVWriter("data/some_boolean.csv", "w")
    writer.writerow(['org', 'repo', '拒绝率',
                        'null文本拒绝率','非null文本拒绝率',
                        'reviewer拒绝率', '非reviewer拒绝率',
                        'bug词汇拒绝率', '非bug词汇拒绝率',
                        'test拒绝率', '非test拒绝率'])
    for org, repo in org_list:
        reviewer_set = set([x['name'] for x in list(client[org]['reviewer'].find())])
        pullinfo_list = client[org]['pullinfo'].find()
        merged, rejected = 0, 0
        null_merged, null_rejected, no_null_merged, no_null_rejected = 0, 0, 0, 0
        reviewer_merged, reviewer_rejected, no_reviewer_merged, no_reviewer_rejected = 0, 0, 0, 0
        bug_merged, bug_rejected, no_bug_merged, no_bug_rejected = 0, 0, 0, 0
        test_merged, test_rejected, no_test_merged, no_test_rejected = 0, 0, 0, 0

        # 构建文件字典
        pulllfile_list = list(client[org]['pullfile'].find())
        file_dict = {}
        for pullfile in pulllfile_list:
            if int(pullfile['number']) in file_dict:
                file_dict[int(pullfile['number'])].append(pullfile['filename'])
            else:
                file_dict[int(pullfile['number'])] = [pullfile['filename']]


        for pullinfo in pullinfo_list:
            print(org+":"+str(pullinfo['number']))
            has_bug_word = False
            has_test = False
            if pullinfo['merged']:
                merged += 1
            else:
                rejected += 1

            # 是否是内部人员
            if pullinfo['author'] in reviewer_set and pullinfo['merged']:
                reviewer_merged += 1
            if pullinfo['author'] in reviewer_set and not pullinfo['merged']:
                reviewer_rejected += 1
            if pullinfo['author'] not in reviewer_set and pullinfo['merged']:
                no_reviewer_merged += 1
            if pullinfo['author'] not in reviewer_set and not pullinfo['merged']:
                no_reviewer_rejected += 1

            # body 是否为空
            if pullinfo['body_token'] == '' and pullinfo['merged']:
                    null_merged += 1
            if pullinfo['body_token'] == '' and not pullinfo['merged']:
                    null_rejected += 1
            if pullinfo['body_token'] != '' and pullinfo['merged']:
                    no_null_merged += 1
            if pullinfo['body_token'] != '' and not pullinfo['merged']:
                    no_null_rejected += 1

            # 是否带有bug词
            for word in str(pullinfo['title_token'] + " " + pullinfo['body_token']).split(" "):
                if word in bug_set:
                    has_bug_word = True
                    break

            if has_bug_word and pullinfo['merged']:
                bug_merged += 1
            if has_bug_word and not pullinfo['merged']:
                bug_rejected += 1
            if not has_bug_word and pullinfo['merged']:
                no_bug_merged += 1
            if not has_bug_word and not pullinfo['merged']:
                no_bug_rejected += 1

            # 是否带有测试文件
            if pullinfo['number'] in file_dict:
                for file in file_dict[pullinfo['number']]:
                    if re.match(".*test.*", file):
                        has_test = True
                        break

            if has_test and pullinfo['merged']:
                test_merged += 1
            if has_test and not pullinfo['merged']:
                test_rejected += 1
            if not has_test and pullinfo['merged']:
                no_test_merged += 1
            if not has_test and not pullinfo['merged']:
                no_test_rejected += 1

        writer.writerow([org, repo,
                            round(rejected/(merged+rejected), 5),
                            round(null_rejected/(null_merged+null_rejected), 5),
                            round(no_null_rejected/(no_null_merged+no_null_rejected), 5),
                            round(reviewer_rejected/(reviewer_rejected+reviewer_merged), 5),
                            round(no_reviewer_rejected/(no_reviewer_rejected+no_reviewer_merged), 5),
                            round(bug_rejected/(bug_rejected+bug_merged), 5),
                            round(no_bug_rejected/(no_bug_rejected+no_bug_merged), 5),
                            round(test_rejected / (test_rejected + test_merged), 5),
                            round(no_test_rejected/(no_test_rejected+no_test_merged), 5)
                         ])

@mongo
def some_boolean_addition(client):
    '''
    是否包含外链
    :param client: 
    :return:'''
    writer = getCSVWriter("data/some_boolean.csv", "a")
    writer.writerow(['org', 'repo',
                     '外链拒绝率', '非外链拒绝率'
                    ])
    for org, repo in org_list:
        pullinfo_list = client[org]['pullinfo'].find()
        forward_merged, forward_rejected, no_forward_merged, no_forward_rejected = 0, 0, 0, 0
        attr_list = client[org]['ayonel'].find()
        attr_dict = {}
        for attr in attr_list:
            attr_dict[int(attr['number'])] = attr

        for pullinfo in pullinfo_list:
            if attr_dict[pullinfo['number']]['forward_link'] and pullinfo['merged']:
                forward_merged += 1
            if attr_dict[pullinfo['number']]['forward_link'] and not pullinfo['merged']:
                forward_rejected += 1
            if not attr_dict[pullinfo['number']]['forward_link'] and pullinfo['merged']:
                no_forward_merged += 1
            if not attr_dict[pullinfo['number']]['forward_link'] and not pullinfo['merged']:
                no_forward_rejected += 1

        writer.writerow([org, repo,
                         round(forward_rejected / (forward_merged + forward_rejected), 5),
                         round(no_forward_rejected / (no_forward_merged + no_forward_rejected), 5)
                         ])
@mongo
def some_number_medium_addition(client):
    '''
    统计接受与不接受的pr之间一些技术变量的差异
    代码比例
    以前对项目提交的pr数
    以前对项目提交的pr通过率
    以前对项目提交的pr评审时间
    项目最近3月通过率
    项目最近3月commit
    项目最近3月issue
    项目最近3月pr
    与最近3月接受pr文本相似度
    与最近3月拒绝pr文本相似度
    与最近3月接受pr文件相似度
    与最近3月拒绝pr文件相似度
    :param client: 
    :return: 
    '''
    writer = getCSVWriter("data/some_number_medium.csv", "a")
    writer.writerow(['org', 'repo',
                     # '接受代码比例长度', '拒绝代码比例长度', '',
                     # '接受-以前对项目提交的pr数', '拒绝-以前对项目提交的pr数', '',
                     # '接受-以前对项目提交的pr通过率', '拒绝-以前对项目提交的pr通过率', '',
                     # '接受-以前对项目提交的pr评审时间', '拒绝-以前对项目提交的pr评审时间', '',
                     # '接受-项目最近3月通过率', '拒绝-项目最近3月通过率', '',
                     # '接受-项目最近3月commit', '拒绝-项目最近3月commit', '',
                     # '接受-项目最近3月pr', '拒绝-项目最近3月pr', '',
                     '接受-与最近3月接受title文本相似度', '拒绝-与最近3月接受title相似度', '',
                     '接受-与最近3月拒绝title文本相似度', '拒绝-与最近3月拒绝title文本相似度', '',
                     '接受-与最近3月接受body文本相似度', '拒绝-与最近3月接受body相似度', '',
                     '接受-与最近3月拒绝body文本相似度', '拒绝-与最近3月拒绝body文本相似度', '',
                     # '接受-与最近3月接受pr文件相似度', '拒绝-与最近3月接受pr文件相似度', '',
                     # '接受-与最近3月拒绝pr文件相似度', '拒绝-与最近3月拒绝pr文件相似度', '',
                     ])
    for org, repo in org_list:
        pullinfo_list = list(client[org]['pullinfo'].find())
        attr_list = client[org]['ayonel'].find()
        attr_dict = {}
        for attr in attr_list:
            attr_dict[int(attr['number'])] = attr
        merged_3merged_title_similarity = []
        rejected_3merged_title_similarity = []
        merged_3rejected_title_similarity = []
        rejected_3rejected_title_similarity = []

        merged_3merged_body_similarity = []
        rejected_3merged_body_similarity = []
        merged_3rejected_body_similarity = []
        rejected_3rejected_body_similarity = []

        # merged_code_proportion_len = []
        # rejected_code_proportion_len = []
        # merged_history_commit_num = []
        # rejected_history_commit_num = []
        # merged_history_commit_passrate = []
        # rejected_history_commit_passrate = []
        # merged_history_commit_review_time = []
        # rejected_history_commit_review_time = []
        # merged_recent_project_passrate = []
        # rejected_recent_project_passrate = []
        # merged_recent_3_month_commit = []
        # rejected_recent_3_month_commit = []
        # merged_recent_3_month_pr = []
        # rejected_recent_3_month_pr = []

        for pullinfo in pullinfo_list:
            if pullinfo['merged']:
                merged_3merged_title_similarity.append(attr_dict[pullinfo['number']]['title_similarity_merged'])
            else:
                rejected_3merged_title_similarity.append(attr_dict[pullinfo['number']]['title_similarity_merged'])

            if pullinfo['merged']:
                merged_3rejected_title_similarity.append(attr_dict[pullinfo['number']]['title_similarity_rejected'])
            else:
                rejected_3rejected_title_similarity.append(attr_dict[pullinfo['number']]['title_similarity_rejected'])

            if pullinfo['merged']:
                merged_3merged_body_similarity.append(attr_dict[pullinfo['number']]['body_similarity_merged'])
            else:
                rejected_3merged_body_similarity.append(attr_dict[pullinfo['number']]['body_similarity_merged'])

            if pullinfo['merged']:
                merged_3rejected_body_similarity.append(attr_dict[pullinfo['number']]['body_similarity_rejected'])
            else:
                rejected_3rejected_body_similarity.append(attr_dict[pullinfo['number']]['body_similarity_rejected'])


                    # if pullinfo['merged']:
            #     merged_code_proportion_len.append(attr_dict[pullinfo['number']]['text_code_proportion'])
            # else:
            #     rejected_code_proportion_len.append(attr_dict[pullinfo['number']]['text_code_proportion'])
        #
        #     if pullinfo['merged']:
        #         merged_history_commit_num.append(attr_dict[pullinfo['number']]['history_commit_passrate'])
        #     else:
        #         rejected_history_commit_num.append(attr_dict[pullinfo['number']]['history_commit_passrate'])
        #
        #     if pullinfo['merged']:
        #         merged_history_commit_passrate.append(attr_dict[pullinfo['number']]['history_commit_passrate'])
        #     else:
        #         rejected_history_commit_passrate.append(attr_dict[pullinfo['number']]['history_commit_passrate'])
        #
        #     if pullinfo['merged']:
        #         merged_history_commit_num.append(attr_dict[pullinfo['number']]['history_commit_passrate'])
        #     else:
        #         rejected_history_commit_num.append(attr_dict[pullinfo['number']]['history_commit_passrate'])
        #
        #     if pullinfo['merged']:
        #         merged_history_commit_review_time.append(attr_dict[pullinfo['number']]['history_commit_review_time'])
        #     else:
        #         rejected_history_commit_review_time.append(attr_dict[pullinfo['number']]['history_commit_review_time'])
        #
        #     if pullinfo['merged']:
        #         merged_recent_project_passrate.append(attr_dict[pullinfo['number']]['recent_project_passrate'])
        #     else:
        #         rejected_recent_project_passrate.append(attr_dict[pullinfo['number']]['recent_project_passrate'])
        #
        #     if pullinfo['merged']:
        #         merged_recent_3_month_commit.append(attr_dict[pullinfo['number']]['recent_3_month_commit'])
        #     else:
        #         rejected_recent_3_month_commit.append(attr_dict[pullinfo['number']]['recent_3_month_commit'])
        #
        #     if pullinfo['merged']:
        #         merged_recent_3_month_pr.append(attr_dict[pullinfo['number']]['recent_3_month_pr'])
        #     else:
        #         rejected_recent_3_month_pr.append(attr_dict[pullinfo['number']]['recent_3_month_pr'])


        merged_3merged_title_similarity_medium = get_medium(merged_3merged_title_similarity)
        rejected_3merged_title_similarity_medium = get_medium(rejected_3merged_title_similarity)
        merged_3rejected_title_similarity_medium = get_medium(merged_3rejected_title_similarity)
        rejected_3rejected_title_similarity_medium = get_medium(rejected_3rejected_title_similarity)
        merged_3merged_body_similarity_medium = get_medium(merged_3merged_body_similarity)
        rejected_3merged_body_similarity_medium = get_medium(rejected_3merged_body_similarity)
        merged_3rejected_body_similarity_medium = get_medium(merged_3rejected_body_similarity)
        rejected_3rejected_body_similarity_medium = get_medium(rejected_3rejected_body_similarity)


        # merged_3merged_text_similarity_medium = get_medium(merged_3merged_text_similarity)
        # rejected_3merged_text_similarity_medium = get_medium(rejected_3merged_text_similarity)
        # merged_3rejected_text_similarity_medium = get_medium(merged_3rejected_text_similarity)
        # rejected_3rejected_text_similarity_medium = get_medium(rejected_3rejected_text_similarity)
        # merged_3merged_file_similarity_medium = get_medium(merged_3merged_file_similarity)
        # rejected_3merged_file_similarity_medium = get_medium(rejected_3merged_file_similarity)
        # merged_3rejected_file_similarity_medium = get_medium(merged_3rejected_file_similarity)
        # rejected_3rejected_file_similarity_medium = get_medium(rejected_3rejected_file_similarity)
        # merged_code_proportion_medium = get_medium(merged_code_proportion_len)
        # rejected_code_proportion_medium = get_medium(rejected_code_proportion_len)
        # merged_history_commit_num_medium = get_medium(merged_history_commit_num)
        # rejected_history_commit_num_medium = get_medium(rejected_history_commit_num)
        # merged_history_commit_passrate_medium = get_medium(merged_history_commit_passrate)
        # rejected_history_commit_passrate_medium =get_medium(rejected_history_commit_passrate)
        # merged_history_commit_review_time_medium = get_medium(merged_history_commit_review_time)
        # rejected_history_commit_review_time_medium = get_medium(rejected_history_commit_review_time)
        # merged_recent_project_passrate_medium = get_medium(merged_recent_project_passrate)
        # rejected_recent_project_passrate_medium = get_medium(rejected_recent_project_passrate)
        # merged_recent_3_month_commit_medium = get_medium(merged_recent_3_month_commit)
        # rejected_recent_3_month_commit_medium =get_medium(rejected_recent_3_month_commit)
        # merged_recent_3_month_pr_medium = get_medium(merged_recent_3_month_pr)
        # rejected_recent_3_month_pr_medium = get_medium(rejected_recent_3_month_pr)


        writer.writerow([
            org,
            repo,
            # merged_code_proportion_medium,rejected_code_proportion_medium,'',
            # merged_history_commit_num_medium,rejected_history_commit_num_medium,'',
            # merged_history_commit_passrate_medium,rejected_history_commit_passrate_medium,'',
            # merged_history_commit_review_time_medium,rejected_history_commit_review_time_medium,'',
            # merged_recent_project_passrate_medium,rejected_recent_project_passrate_medium,'',
            # merged_recent_3_month_commit_medium,rejected_recent_3_month_commit_medium,'',
            # merged_recent_3_month_pr_medium,rejected_recent_3_month_pr_medium,'',
            merged_3merged_title_similarity_medium, rejected_3merged_title_similarity_medium,'',
            merged_3rejected_title_similarity_medium, rejected_3rejected_title_similarity_medium,'',
            merged_3merged_body_similarity_medium, rejected_3merged_body_similarity_medium,'',
            merged_3rejected_body_similarity_medium, rejected_3rejected_body_similarity_medium,'',
            ])



if __name__ == '__main__':
    # 除去自己关闭的pr之后的项目拒绝率
    # remove_self_close()
    # some_number_medium()
    # some_boolean()
    some_number_medium_addition()
    some_boolean_addition()