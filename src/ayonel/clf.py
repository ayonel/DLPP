'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com

 特征选择方法：
 先在ayonel_numerical_attr中按照单决策树 选取了前10个重要特征

 在之后的选择中，以这10个特征为基础，逐个加入剩余特征，并输出平均准确率。

 选择提升最多的一个特征，并将这个特征纳入基础特征集合，再逐个加入剩余特征，直到准确率无提升。


'''
import numpy as np
import csv
from src.ayonel.LoadData import *
from src.constants import *
from src.eval.eval_utils import precision_recall_f1
from src.utils import *

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
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
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import train_test_split as tts
from imblearn.combine import SMOTEENN
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
from costcla.models import CostSensitiveRandomForestClassifier
from costcla.models import CostSensitiveBaggingClassifier
from sklearn.metrics import roc_auc_score

SEG_PROPORTION = 8/10
FOLDS = 5
# 按重要性排序之后的
ayonel_numerical_attr = [
    'last_10_pr_rejected',
    'history_pass_pr_num',
    'last_10_pr_merged',
    'commits',
    'history_commit_num',
    'files_changes',
    'inline_comment_num',
    'commits_files_touched',
    'history_commit_passrate',

    # 后来补充
    'src_addition',
    'pr_file_rejected_count',
    'team_size',
    'src_deletion',  # 降
    'pr_file_rejected_proportion',  # 降
    'src_churn',
    'text_code_proportion',
    'pr_file_merged_count_decay',
    'history_pr_num_decay',  # 降
    'pr_file_submitted_count_decay',

    # 后来补充稍差
    # 'history_commit_review_time',
    # 'recent_1_month_project_pr_num',  # 降
    # 'pr_file_merged_count',  # 降 17
    # 'pr_file_submitted_count',  # 降
    # 'recent_3_month_project_pr_num',  # 降
    # 'recent_3_month_pr',  # 降 21
    # 'pr_file_merged_proportion',  # 降
    # 'recent_3_month_commit',
    # 'recent_project_passrate',  # 降26


    # 太差
    # 'pr_file_rejected_count_decay',
    # 'perc_ext_contribs',  # 降 29
    # 'body_similarity_merged',
    # 'title_similarity_rejected', # 降32
    # 'recent_3_month_project_pr_num_decay', # 降34
    # 'title_similarity_merged',
    # 'last_10_pr_merged_decay',# 降
    # 'recent_1_month_project_pr_num_decay',# 降
    # 'body_similarity_rejected',# 降 38
    # 'file_similarity_merged',
    # 'text_similarity_merged',
    # 'file_similarity_rejected',# 降 38
    # 'text_similarity_rejected',# 降 38
    # 'last_10_pr_rejected_decay',

]


ayonel_boolean_attr = [
    'is_reviewer_commit',
    # 'has_test',
    'text_forward_link',
    'last_pr',
    # 'has_body',
]

ayonel_categorical_attr_handler = [
    ('week', week_handler)
]


def train(clf, X, y):
    clf.fit(X, y)


@mongo
def run(client, clf, print_prf=False, print_main_proportion=False):
    attr_dict, label_dict, pullinfo_list_dict = load_data(ayonel_numerical_attr=ayonel_numerical_attr, ayonel_boolean_attr=ayonel_boolean_attr,
                                  ayonel_categorical_attr_handler=ayonel_categorical_attr_handler)
    ACC = 0.0
    for org, repo in org_list:
        input_X = attr_dict[org]
        input_y = label_dict[org]
        seg_point = int(len(input_X) * SEG_PROPORTION)

        train_X = np.array(input_X[:seg_point])
        train_y = np.array(input_y[:seg_point])
        # X_sparse = coo_matrix(train_X)
        #
        # train_X, X_sparse, train_y = shuffle(train_X, X_sparse, train_y, random_state=0)
        # train_X, train_y = AS().fit_sample(train_X, train_y)
        test_X = np.array(input_X[seg_point:])
        test_y = np.array(input_y[seg_point:])

        train(clf, train_X, train_y)
        accuracy = clf.score(test_X, test_y)
        ACC += accuracy

        predict_result = clf.predict(test_X).tolist()
        actual_result = test_y.tolist()
        precision, recall, F1 = precision_recall_f1(predict_result, actual_result)
        print(accuracy, end='')
        if print_prf:
            print(",%f,%f,%f" % (precision, recall, F1), end='')
        if print_main_proportion:
            main_proportion = predict_result.count(1) / len(predict_result)
            print(',%f' % (main_proportion if main_proportion > 0.5 else 1 - main_proportion), end='')
        print()
    print(ACC/len(org_list))


# 按月训练
@mongo
def run_monthly(client, model, deserialize=False, print_prf_each=False, print_main_proportion=False, print_AUC=False, MonthGAP=1, persistence=False):
    data_dict, pullinfo_list_dict = load_data_monthly(ayonel_numerical_attr=ayonel_numerical_attr, ayonel_boolean_attr=ayonel_boolean_attr,
                                  ayonel_categorical_attr_handler=ayonel_categorical_attr_handler, MonthGAP=MonthGAP)

    for org, repo in org_list:
        print(org+",", end='')
        pullinfo_list = pullinfo_list_dict[org]
        batch_iter = data_dict[org]
        train_batch = batch_iter.__next__()
        train_X = np.array(train_batch[0])
        train_y = np.array(train_batch[1])
        cursor = train_y.size  # 游标，用于记录第一条开始预测pr的位置
        predict_result = []
        predict_result_prob = []
        actual_result = []
        mean_accuracy = 0

        round = 1
        for batch in batch_iter:
            if len(batch[0]) == 0:  # 测试集没有数据，直接预测下一batch
                continue
            test_X = np.array(batch[0])
            test_y = np.array(batch[1])
            # X_sparse = coo_matrix(train_X)
            # train_X, X_sparse, train_y = shuffle(train_X, X_sparse, train_y, random_state=0)

            clf = None
            if deserialize:
                if model == "xgboost":
                    parameters = client[org]['model'].find_one({'model': 'xgboost', 'round':round, 'gap': MonthGAP}, {'_id': 0, 'model': 0, 'gap': 0, 'round': 0})
                    clf = XGBClassifier(seed=RANDOM_SEED, **parameters)
                    train(clf, train_X, train_y)
                elif model == 'randomforest':
                    parameters = client[org]['model'].find_one({'model': 'randomforest', 'round': round, 'gap': MonthGAP}, {'_id': 0, 'model': 0, 'gap': 0, 'round': 0})
                    clf = RandomForestClassifier(random_state=RANDOM_SEED, **parameters)
                    train(clf, train_X, train_y)
                else:
                    raise RuntimeError("请指定分类器")
            else:
                if model == "xgboost":
                    clf = XGBClassifier(seed=RANDOM_SEED)
                    train(clf, train_X, train_y)
                elif model == 'randomforest':
                    clf = RandomForestClassifier(random_state=RANDOM_SEED)
                    train(clf, train_X, train_y)
                elif model == 'costsensitiverandomforest':
                    clf = CostSensitiveBaggingClassifier()
                    cost_mat = []
                    T_count = 0
                    F_count = 0
                    for label in train_y:
                        if label == 1:
                            T_count += 1
                        else:
                            F_count += 1

                    for i in range(len(train_y)):
                        cost_mat.append([0.0,0.0,0.0,0.0])

                    # cost_mat = np.zeros((len(train_y), 4))
                    # cost_mat[:, 0] = 0.0,                   # false_positive_cost
                    # cost_mat[:, 1] = 0.0                    # false_positive_cost
                    # cost_mat[:, 2] = 0.0                    # true_positive_cost
                    # cost_mat[:, 3] = 0.0,                   # true_negative_cost

                    clf.fit(train_X, train_y, np.array(cost_mat))
                else:
                    raise RuntimeError("请指定分类器")

            # print(clf.score(test_X, test_y))

            # # # 过采样
            # if train_y.tolist().count(0) <= 6 or train_y.tolist().count(1) <= 6:
            #     train(clf, train_X, train_y)
            # else:
            #     resample_train_X, resample_train_y = SMOTE(ratio='auto', random_state=RANDOM_SEED).fit_sample(train_X, train_y)
            #     train(clf, resample_train_X, resample_train_y)
            actual_result += test_y.tolist()  # 真实结果
            predict_result += clf.predict(test_X).tolist()  # 预测结果
            predict_result_prob += [x[0] for x in clf.predict_proba(test_X).tolist()]
            mean_accuracy += clf.score(test_X, test_y)
            train_X = np.concatenate((train_X, test_X))
            train_y = np.concatenate((train_y, test_y))
            round += 1

        acc_num = 0
        for i in range(len(actual_result)):
            if actual_result[i] == predict_result[i]:
                acc_num += 1
        # 需要将结果入库
        if persistence:
            for i in range(len(predict_result)):
                number = int(pullinfo_list[cursor + i]['number'])
                data = {
                    'number':           number,
                    'org':              org,
                    'repo':             repo,
                    'created_at':       float(pullinfo_list[cursor + i]['created_at']),
                    'closed_at':        float(pullinfo_list[cursor + i]['closed_at']),
                    'title':            pullinfo_list[cursor + i]['title'],
                    'submitted_by':     pullinfo_list[cursor + i]['author'],
                    'merged':           pullinfo_list[cursor + i]['merged'],
                    'predict_merged':   True if predict_result[i] == 0 else False
                }

                client[persistence_db][persistence_col].insert(data)

        print(acc_num / len(actual_result), end='')
        # print(predict_result)
        # print(actual_result)

        if print_prf_each:
            merged_precision, merged_recall, merged_F1 = precision_recall_f1(predict_result, actual_result, POSITIVE=0)
            rejected_precision, rejected_recall, rejected_F1 = precision_recall_f1(predict_result, actual_result, POSITIVE=1)
            print(',%f,%f,%f,%f,%f,%f' % (merged_F1, merged_precision, merged_recall, rejected_F1,rejected_precision, rejected_recall ), end='')


        if print_main_proportion:
            main_proportion = predict_result.count(1) / len(predict_result)
            print(',%f' % (main_proportion if main_proportion > 0.5 else 1 - main_proportion), end='')

        if print_AUC:
            y = np.array(actual_result)
            pred = np.array(predict_result_prob)
            fpr, tpr, thresholds = roc_curve(y, pred)
            AUC = auc(fpr, tpr)
            print(',%f' % (AUC if AUC > 0.5 else 1 - AUC), end='')
        print()


if __name__ == '__main__':

    # model = 'costsensitiverandomforest'
    model = 'xgboost'
    # model = 'randomforest'
    # clf = RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced_subsample')
    # clf = CostSensitiveBaggingClassifier()
    run_monthly(model, deserialize=True, print_prf_each=True, print_main_proportion=False, print_AUC=True, MonthGAP=6, persistence=False)

    # run(XGBClassifier(seed=RANDOM_SEED))

