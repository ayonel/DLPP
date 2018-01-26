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
import xgboost as xgb
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
# ayonel_numerical_attr = [
#     'history_commit_passrate',
#     'commits_files_touched',
#     # 'last_10_pr_rejected',
#     'commits',
#     'files_changes',
#
#     'last_10_pr_merged',
#     'src_churn',
#     'pr_file_rejected_proportion',
#     'src_addition',
#     'history_pr_num_decay',
#     'team_size',
#     'src_deletion',
#
#
#     # # 淘汰
#     # 'pr_file_merged_count_decay',
#     # 'pr_file_submitted_count_decay'
#     # 'pr_file_rejected_count',
#     # 'src_deletion',
#     # 'src_churn',
#     # 'text_code_proportion',
#     # 'history_commit_review_time',
#     # 'recent_1_month_project_pr_num',
#     # 'pr_file_merged_count',
#     # 'pr_file_submitted_count',
#     # 'recent_3_month_project_pr_num',
#     # 'recent_3_month_pr',
#     # 'pr_file_merged_proportion',
#     # 'recent_3_month_commit',
#     # 'recent_project_passrate',
#
#     # 太差
#     # 'pr_file_rejected_count_decay',
#     # 'perc_ext_contribs',  # 降 29
#     # 'body_similarity_merged',
#     # 'title_similarity_rejected', # 降32
#     # 'recent_3_month_project_pr_num_decay', # 降34
#     # 'title_similarity_merged',
#     # 'last_10_pr_merged_decay',# 降
#     # 'recent_1_month_project_pr_num_decay',# 降
#     # 'body_similarity_rejected',# 降 38
#     # 'file_similarity_merged',
#     # 'text_similarity_merged',
#     # 'file_similarity_rejected',# 降 38
#     # 'text_similarity_rejected',# 降 38
#     # 'last_10_pr_rejected_decay',
#
# ]

ayonel_numerical_attr = [
    'history_commit_passrate',
    'commits_files_touched',
    'last_10_pr_rejected',
    'src_churn',
    'pr_file_rejected_proportion',
    # 'is_reviewer_commit',
    'commits',
    'text_similarity_merged',
    # 'last_pr',
    'text_similarity_rejected',
    # 'has_test',
    'src_deletion',
    # 'has_body',
    'files_changes',
    'src_addition',
    'last_10_pr_merged',
    # 'text_forward_link',

    ## 以下属性无用
    # 'history_pass_pr_num',
    # 'history_commit_review_time',
    # 'history_commit_num',
    # 'pr_file_merged_proportion',
    # 'pr_file_rejected_count',
    # 'pr_file_merged_count',
    # 'pr_file_submitted_count',
    # 'test_churn',
    # 'sloc',
    # 'recent_3_month_project_pr_num',
    # 'text_code_proportion',
    # 'team_size',
    # 'file_similarity_rejected',
    # 'file_similarity_merged',
    # 'recent_project_passrate',
    # 'test_lines_per_kloc',
    # 'recent_3_month_commit',
    # 'perc_ext_contribs',
]

# 5个bool属性都用了
ayonel_boolean_attr = [
    'is_reviewer_commit',
    'last_pr',
    'has_test',
    'has_body',
    'text_forward_link',

]

ayonel_categorical_attr_handler = [
    # ('week', week_handler)
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
def run_monthly(client, clf, thred=0.5, deserialize=False, over_sample=False, print_prf_each=False, print_main_proportion=False, print_AUC=False, MonthGAP=1, persistence=False):
    data_dict, pullinfo_list_dict = load_data_monthly(ayonel_numerical_attr=ayonel_numerical_attr, ayonel_boolean_attr=ayonel_boolean_attr,
                                  ayonel_categorical_attr_handler=ayonel_categorical_attr_handler, MonthGAP=MonthGAP)

    # 构造代价矩阵
    def _cal_cost_mat(train_y):
        cost_mat = []
        T_count = 0
        F_count = 0
        for label in train_y:
            if label == 1:
                T_count += 1
            else:
                F_count += 1

        for i in range(len(train_y)):
            cost_mat.append([0.0, 0.0, 0.0, 0.0])
        return np.array(cost_mat)

    # 初始化分类器
    def _init_classifier(models, deserialize, org, round, MonthGAP):
        clf_list = []
        for model_name, weight in models.items():
            if deserialize:
                if model_name == "xgboost":  # 'scale_pos_weight'此参数不要用于初始化模型
                    parameters = client[org]['model'].find_one({'model': 'xgboost', 'round': round, 'gap': MonthGAP},
                                                               {'_id': 0, 'model': 0, 'gap': 0, 'round': 0, 'scale_pos_weight':0})
                    clf = XGBClassifier(seed=RANDOM_SEED, **parameters)
                    clf_list.append((clf, weight))
                elif model_name == 'randomforest':
                    parameters = client[org]['model'].find_one({'model': 'randomforest', 'round': round, 'gap': MonthGAP},
                                                               {'_id': 0, 'model': 0, 'gap': 0, 'round': 0})
                    clf = RandomForestClassifier(random_state=RANDOM_SEED, **parameters)
                    clf_list.append((clf, weight))
                else:
                    raise RuntimeError("请指定分类器")
            else:
                if model_name == "xgboost":
                    clf = XGBClassifier(seed=RANDOM_SEED)
                    clf_list.append((clf, weight))
                elif model_name == 'randomforest':
                    clf = RandomForestClassifier(random_state=RANDOM_SEED)
                    clf_list.append((clf, weight))
                elif model_name == 'costsensitiverandomforest':
                    clf = CostSensitiveBaggingClassifier()
                    clf_list.append((clf, weight))
                elif model_name == 'svm':
                    clf = SVC(probability=True)
                    clf_list.append((clf, weight))
                elif model_name == 'bayes':
                    clf = GaussianNB()
                    clf_list.append((clf, weight))
                else:
                    raise RuntimeError("请指定分类器")
                clf_list.append((clf, weight))

            if clf_list == [] or clf_list.__contains__(None):
                raise RuntimeError("分类器初始化失败")
        return clf_list

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
            # clf_list = _init_classifier(models, deserialize, org, round, MonthGAP)
            # print(clf.get_params())
            ########################################SMOTE过采样#####################################
            # for clf, weight in clf_list:
            #     if over_sample:
            #         if clf.__class__.__name__ == 'CostSensitiveRandomForestClassifier':
            #             if train_y.tolist().count(0) <= 6 or train_y.tolist().count(1) <= 6:
            #                 # cost_mat = np.zeros((len(train_y), 4))
            #                 # cost_mat[:, 0] = 0.0,                   # false_positive_cost
            #                 # cost_mat[:, 1] = 0.0                    # false_positive_cost
            #                 # cost_mat[:, 2] = 0.0                    # true_positive_cost
            #                 # cost_mat[:, 3] = 0.0,                   # true_negative_cost
            #                 clf.fit(train_X, train_y, _cal_cost_mat(train_y))
            #             else:
            #                 resample_train_X, resample_train_y = SMOTE(ratio='auto', random_state=RANDOM_SEED).fit_sample(
            #                     train_X, train_y)
            #                 clf.fit(resample_train_X, resample_train_y, _cal_cost_mat(resample_train_y))
            #         else:
            #             if train_y.tolist().count(0) <= 6 or train_y.tolist().count(1) <= 6:
            #                 train(clf, train_X, train_y)
            #             else:
            #                 resample_train_X, resample_train_y = SMOTE(ratio='auto', random_state=RANDOM_SEED).fit_sample(
            #                     train_X, train_y)
            #                 train(clf, resample_train_X, resample_train_y)
            #
            #     else:   # 正常
            #         train(clf, train_X, train_y)
            #     batch_predict_result_proba += clf.predict_proba(test_X)[:, 0]
            train(clf, train_X, train_y)
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

    # clf = RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced_subsample')
    # clf = CostSensitiveBaggingClassifier()
    clf = XGBClassifier(seed=RANDOM_SEED)
    run_monthly(clf,
                thred=0.5,
                deserialize=False,
                over_sample=True,
                print_prf_each=False,
                print_main_proportion=False,
                print_AUC=True,
                MonthGAP=6,
                persistence=False)

    # run(XGBClassifier(seed=RANDOM_SEED))

