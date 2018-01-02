'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
'''

from src.constants import *
from src.database.dbutil import *
import pymongo
import re
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier



@mongo
def main(client):

    total_feature_score = {}

    for org, repo in org_list:
        attr_list = list(client[org]['ayonel'].find())
        attr_list.sort(key=lambda x:int(x['number']))

        attrinfo_list = [
            ("is_reviewer_commit",True),
            ("has_body", True),
            ("text_forward_link", True),
            ("has_test", True),
            ("files_changes", False),
            ("recent_project_passrate" ,False),
            ("history_commit_num" ,False),
            ("history_commit_passrate", False),
            # ("is_approved" ,False),
            ("src_churn",False),
            ("commits",False),
            ("recent_3_month_pr" ,False),
            ("history_commit_review_time",False),
            ("src_addition" ,False),
            ("src_deletion" ,False),
            ("recent_3_month_commit" ,False),
            ("text_code_proportion" ,False),
            ("text_similarity_merged",False),
            ("title_similarity_rejected",False),
            ("body_similarity_rejected" ,False),
            ("file_similarity_rejected" ,False),
            ("text_similarity_rejected" ,False),
            ("file_similarity_merged" ,False),
            ("title_similarity_merged",False),
            ("body_similarity_merged", False),
        ]

        pullinfo_list = client[org]['pullinfo'].find()
        label_dict = {}
        for pullinfo in pullinfo_list:
            label_dict[pullinfo['number']] = True if pullinfo['merged'] else False

        X = []
        y = []
        for attr in attr_list:
            tmp = []
            for attrinfo in attrinfo_list:
                if attrinfo[1]: # 是boolean
                    if attr[attrinfo[0]]:
                        tmp.append(1.0)
                    else:
                        tmp.append(0.0)
                else:
                    tmp.append(round(attr[attrinfo[0]],2))

            X.append(tmp)
            y.append(1 if label_dict[int(attr['number'])] else 0)


        rf = RandomForestClassifier()
        rf.fit(X, y)

        L = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), [x[0] for x in attrinfo_list]),
               reverse=True)
        for feature in L:
            if feature[1] not in total_feature_score:
                total_feature_score[feature[1]] = float(feature[0])
            else:
                total_feature_score[feature[1]] += float(feature[0])
        print(org+":over")
    total_feature_score_list = []
    for feature in total_feature_score:
        total_feature_score_list.append((feature, total_feature_score[feature]))

    print(total_feature_score_list)
    total_feature_score_list.sort(key=lambda x: float(x[1]), reverse=True)

    outfile = open(ROOT_PATH+"/src/stat/attr_analyse/data/特征重要性排序.txt", "w")
    for feature in total_feature_score_list:
        outfile.write(str(feature)+"\n")


if __name__ == '__main__':
    client = get_connection()
    main(client)