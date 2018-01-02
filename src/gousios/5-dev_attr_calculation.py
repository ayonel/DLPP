# 计算gousios论文中的developer属性
import pymongo
import src.database.dbutil as dbutil
import src.constants as constants


# 历史集中由该pr的提交者提交的pr数量（merged与closed）以及历史集中该pr的提交提交的pr的通过率【0~1】
def prev_pullreqs_AND_requester_succ_rate(pullinfo_list, D, result_dict):
    for pullinfo in pullinfo_list:
        history_pull_count = 0
        history_merged_pull_count = 0
        this_create_time = pullinfo['created_at']
        for history_pullinfo in D[pullinfo['author']]:
            if history_pullinfo['created_at'] < this_create_time:
                history_pull_count += 1
                if history_pullinfo['merged'] and history_pullinfo['merged_at'] < this_create_time:
                    history_merged_pull_count += 1

        result_dict[pullinfo['number']]['prev_pullreqs'] = history_pull_count
        if history_pull_count == 0:
            result_dict[pullinfo['number']]['requester_succ_rate'] = 0.0
        else:
            result_dict[pullinfo['number']]['requester_succ_rate'] = history_merged_pull_count/history_pull_count
    return result_dict


# 构造空结果字典
def build_empty_result_dict(pullinfo_list):
    result_dict = {}
    for pullinfo in pullinfo_list:
        number = pullinfo['number']
        result_dict[number] = {}
        result_dict[number]['requester_succ_rate'] = 0.0
        result_dict[number]['prev_pullreqs'] = 0
    return result_dict

if __name__ == '__main__':
    client = dbutil.get_connection()
    for org, repo in constants.org_list:
        print(org)
        # 先建立数据库连接
        db = client[org]
        #  pull_dict
        pullinfo_list = list(db['pullinfo'].find().sort('number', pymongo.ASCENDING))

        # 建立以developer为键。pr为值得字典。
        D = {}
        for pullinfo in pullinfo_list:
            if pullinfo['author'] not in D:
                D[pullinfo['author']] = []
                D[pullinfo['author']].append(pullinfo)
            else:
                D[pullinfo['author']].append(pullinfo)

        result_dict = build_empty_result_dict(pullinfo_list)
        result_dict = prev_pullreqs_AND_requester_succ_rate(pullinfo_list, D, result_dict)
        print('prev_pullreqs 以及 requester_succ_rate计算完毕')

        # 开始输出
        for result in result_dict:
            client[org]['gousios'].update({'number': str(result)}, {'$set': result_dict[result]})





