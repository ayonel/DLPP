'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
 description:将gousios的commits_files_touched，perc_ext_contribs迁移到ayonel
'''

from src.database.dbutil import *
from src.constants import *

@mongo
def migerate(client):
    for org, repo in org_list:
        print(org)
        gousios_attr = list(client[org]['gousios'].find({}, {'_id': 0}))
        for attr in gousios_attr:
            data = {
                # 'commits_files_touched': attr['commits_files_touched'],
                # 'perc_ext_contribs': attr['perc_ext_contribs'],
                # 'team_size': attr['team_size'],
                'test_churn': attr['test_churn'],
                'sloc': attr['sloc'],
                'test_lines_per_kloc': attr['test_lines_per_kloc'],
            }
            client[org]['ayonel'].update({'number': attr['number']}, {'$set': data}, upsert=True)

if __name__ == '__main__':
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.ensemble import RandomForestClassifier
    # from xgboost import XGBClassifier
    # from costcla.models import CostSensitiveRandomForestClassifier
    # import numpy as np
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([1, 1, 0, 1])
    # # est = XGBClassifier()
    # # params = {
    # #     'learning_rate': [0.5, 0, 1]
    # # }
    # # clf = Gr idSearchCV(
    # #     estimator=est,
    # #     param_grid=params,
    # #     scoring="accuracy",
    # #     cv=2
    # # )
    # # clf.fit(X, y)
    # clf = CostSensitiveRandomForestClassifier()
    # clf.fit(X, y, np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
    #
    # # clf = RandomForestClassifier()
    # # clf.fit(X,y)
    # print(clf.predict(np.array([[0,0],[1,0]])))

    # for org, repo in org_list:
    #     print("mongoimport -d " + org +" -c model --upsert " + org)

    migerate()


