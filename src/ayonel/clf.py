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
import time
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
from sklearn.naive_bayes import MultinomialNB
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

# 星期几，该分类属性无用
ayonel_categorical_attr_handler = [
    # ('week', week_handler)
]


def train(clf, X, y):
    clf.fit(X, y)


@mongo
def run(client, clf, print_prf=False, print_main_proportion=False):
    attr_dict, label_dict, pullinfo_list_dict = load_data(ayonel_numerical_attr=ayonel_numerical_attr,
                                                          ayonel_boolean_attr=ayonel_boolean_attr,
                                                          ayonel_categorical_attr_handler=ayonel_categorical_attr_handler)
    ACC = 0.0
    for org, repo in org_list:
        input_X = attr_dict[org]
        input_y = label_dict[org]
        seg_point = int(len(input_X) * SEG_PROPORTION)

        train_X = np.array(input_X[:seg_point])
        train_y = np.array(input_y[:seg_point])

        # 随机打乱训练集
        # train_X, X_sparse, train_y = shuffle(train_X, X_sparse, train_y, random_state=0)
        # train_X, train_y = AS()
        # .fit_sample(train_X, train_y)
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


# 按n个月训练
@mongo
def run_monthly(client, clf, print_time=True, over_sample=False, print_acc=False, print_prf_each=False, print_main_proportion=False, print_AUC=False, MonthGAP=1, persistence=False):
    data_dict, pullinfo_list_dict = load_data_monthly(ayonel_numerical_attr=ayonel_numerical_attr, ayonel_boolean_attr=ayonel_boolean_attr,
                                  ayonel_categorical_attr_handler=ayonel_categorical_attr_handler, MonthGAP=MonthGAP)

    for org, repo in org_list:

        train_cost_time = 0     # 训练耗时
        test_cost_time = 0      # 预测耗时
        total_start_time = time.time()  # 每个项目开始时间
        print(org, end='')

        pullinfo_list = pullinfo_list_dict[org]

        batch_iter = data_dict[org]     # 获取数据集迭代器

        train_batch = batch_iter.__next__()  # 获取一轮数据

        train_X = np.array(train_batch[0])
        train_y = np.array(train_batch[1])

        cursor = train_y.size  # 游标，用于记录第一条开始预测pr的位置

        predict_result = []         # 预测结果，0|1表示
        predict_result_prob = []    # 预测结果，概率值
        actual_result = []          # 实际结果，0|1表示
        mean_accuracy = 0           # 平均acc
        round = 1                   # 轮次计数

        for batch in batch_iter:
            if len(batch[0]) == 0:  # 测试集没有数据，直接预测下一batch
                continue
            test_X = np.array(batch[0])
            test_y = np.array(batch[1])

            # 过采样
            if over_sample:
                if train_y.tolist().count(0) <= 6 or train_y.tolist().count(1) <= 6:
                    train(clf, train_X, train_y)
                else:
                    resample_train_X, resample_train_y = SMOTE(ratio='auto', random_state=RANDOM_SEED).fit_sample(
                                train_X, train_y)
                    train(clf, resample_train_X, resample_train_y)
            else:  # 正常算
                if train_y.sum() != 0 and train_y.sum() != train_y.size:
                    train_start_time = time.time()
                    train(clf, train_X, train_y)
                    train_cost_time += time.time() - train_start_time  # 更新训练时间
                else:
                    train_X = np.concatenate((train_X, test_X))         # 将测试集加入训练集中
                    train_y = np.concatenate((train_y, test_y))         # 将测试集加入训练集中
                    continue

            actual_result += test_y.tolist()  # 真实结果
            test_start_time = time.time()
            predict_result += clf.predict(test_X).tolist()  # 预测结果
            test_cost_time += time.time() - test_start_time     # 更新预测时间

            predict_result_prob += [x[0] for x in clf.predict_proba(test_X).tolist()]

            mean_accuracy += clf.score(test_X, test_y)

            train_X = np.concatenate((train_X, test_X))     # 将测试集加入训练集中
            train_y = np.concatenate((train_y, test_y))     # 将测试集加入训练集中
            round += 1

        total_cost_time = time.time() - total_start_time

        acc_num = 0  # 预测正确的pr数量
        for i in range(len(actual_result)):
            if actual_result[i] == predict_result[i]:
                acc_num += 1
        # 如果需要将结果入库
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

                client[persistence_db][persistence_col].insert(data)  # 该表为原型系统使用表

        if print_acc:   # 如果需要打印acc
            print(',%f' % (acc_num / len(actual_result)), end='')

        if print_AUC:   # 如果需要打印AUC
            y = np.array(actual_result)
            pred = np.array(predict_result_prob)
            fpr, tpr, thresholds = roc_curve(y, pred)
            AUC = auc(fpr, tpr)
            print(',%f' % (AUC if AUC > 0.5 else 1 - AUC), end='')

        if print_prf_each:  # 如果需要按照拒绝类、接受类打印精准率、召回率、F1
            merged_precision, merged_recall, merged_F1 = precision_recall_f1(predict_result, actual_result, POSITIVE=0)
            rejected_precision, rejected_recall, rejected_F1 = precision_recall_f1(predict_result, actual_result, POSITIVE=1)
            print(',%f,%f,%f,%f,%f,%f' % (merged_F1, merged_precision, merged_recall, rejected_F1,rejected_precision, rejected_recall ), end='')

        if print_main_proportion:   # 如果需要打印主类占比
            main_proportion = predict_result.count(1) / len(predict_result)
            print(',%f' % (main_proportion if main_proportion > 0.5 else 1 - main_proportion), end='')

        if print_time:      # 如果需要打印时间统计
            print(',%f,%f,%f' % (train_cost_time, test_cost_time, total_cost_time), end='')
        print()


if __name__ == '__main__':
    # clf = SVC(probability=True, kernel='sigmoid')
    # clf = MultinomialNB()
    # clf = RandomForestClassifier(random_state=RANDOM_SEED)
    # clf = CostSensitiveBaggingClassifier()
    clf = XGBClassifier(seed=RANDOM_SEED)       # 定义算法
    run_monthly(clf,
                print_time=False,               # 如果需要打印时间统计
                over_sample=False,              # 如果需要过采样
                print_acc=True,                 # 如果需要打印acc
                print_prf_each=True,            # 如果需要按照拒绝类、接受类打印精准率、召回率、F1
                print_main_proportion=False,    # 如果需要打印主类占比
                print_AUC=True,                 # 如果需要打印AUC
                MonthGAP=6,                     # 以几月为训练间隔
                persistence=False)              # 如果需要入库

