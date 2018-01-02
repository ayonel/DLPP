import numpy as np

from src.database.dbutil import *
from src.ayonel.LoadData import *
from src.constants import *
from src.utils import *


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

SEG_PROPORTION = 8/10
FOLDS = 5

ayonel_numerical_attr = [
    'history_commit_passrate',
    'history_commit_num',
    'commits',
    'files_changes',
    'src_churn',
    'file_similarity_merged',
    # 'src_addition',
    'recent_project_passrate',
    'body_similarity_merged',
    # 'src_deletion',
    'title_similarity_merged',
    'text_similarity_merged',
    'history_commit_review_time',
    'recent_3_month_commit',
    'recent_3_month_pr',
    'text_similarity_rejected',
    'title_similarity_rejected',
    'body_similarity_rejected',
    'file_similarity_rejected'
]


ayonel_categorical_attr = [
    'is_reviewer_commit',
    'has_test',
    'text_forward_link'
    # 'has_body',

]

attr_dict, label_dict = load_data(ayonel_numerical_attr=ayonel_numerical_attr, ayonel_categorical_attr=ayonel_categorical_attr)
def train(clf, X, y):
    clf.fit(X, y)


if __name__ == '__main__':
    client = get_connection()
    for org, repo in org_list:
        input_X = attr_dict[org]
        input_y = label_dict[org]
        seg_point = int(len(attr_dict[org])*SEG_PROPORTION)

        train_X = np.array(attr_dict[org][:seg_point])
        train_y = np.array(label_dict[org][:seg_point])

        test_X = np.array(attr_dict[org][seg_point:])
        test_y = np.array(label_dict[org][seg_point:])
        #########################################随机森林##################################
        # estimator = RandomForestClassifier(random_state=RANDOM_SEED)
        # parameters = {
            # "criterion": ["gini", "entropy"]
            # "max_features": ["auto", "sqrt", "log2"],
            # "min_weight_fraction_leaf": iandfrange(0, 0.501, 0.05),
            # "bootstrap": [True, False],
            # "oob_score": [True, False]
        # }

        ##########################################xgboost##################################
        estimator = XGBClassifier(seed=RANDOM_SEED)
        clf = XGBClassifier(seed=RANDOM_SEED)
        parameters = {
                # "eta": iandfrange(0, 1, 0.05),
                # "gamma": iandfrange(0, 1, 0.1) + iandfrange(1, 10, 1),
                # "objective": ["binary:logistic", "rank:pairwise"]
                # "max_depth": range(1, len(input_X[0]) + 1)
                #  "min_child_weight": range(1, len(input_X[0]) + 1),
                # "subsample": iandfrange(0, 1.01, 0.1)
                # "colsample_bytree": iandfrange(0.1, 1.01, 0.1),
                # "scale_pos_weight": [np.sum(test_y == 0) / np.sum(test_y == 1)]
        }

        ##########################################ExtraTreesClassifier##################################
        # estimator = AdaBoostClassifier(random_state=RANDOM_SEED)
        # parameters = {
            # "learning_rate": iandfrange(0.01, 1.0, 0.05),
            # "n_estimators": iandfrange(10, 100, 5),
            # "algorithm": ['SAMME', 'SAMME.R']
        # }

        ###############################################################################

        # clf = GridSearchCV(
        #         estimator=estimator,
        #         param_grid=parameters,
        #         scoring="accuracy",
        #         cv=TimeSeriesSplit(n_splits=FOLDS)
        # )

        train(clf, train_X, train_y)
        accuracy = clf.score(test_X, test_y)
        # client[org]['model'].update({'model': clf.estimator.__class__.__name__}, {'$set': clf.best_params_}, upsert=True)
        # print(repo)
        # print(clf.best_params_)

        # # client[org]['result'].insert({clf.__class__.__name__: accuracy})
        print(accuracy)






