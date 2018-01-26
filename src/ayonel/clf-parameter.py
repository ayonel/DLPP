'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
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
from xgboost.sklearn import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import train_test_split as tts
from imblearn.combine import SMOTEENN
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
from costcla.models import CostSensitiveRandomForestClassifier
from costcla.models import CostSensitiveBaggingClassifier

SEG_PROPORTION = 8/10
FOLDS = 5
# 按单特征分类器重要性排序之后的

attrs = [
    ('history_commit_passrate', 0),
    ('history_pass_pr_num', 0),
    ('history_commit_review_time', 0),
    ('is_reviewer_commit', 1),
    ('history_commit_num', 0),
    ('last_10_pr_rejected', 0),
    ('commits_files_touched', 0),
    ('last_10_pr_merged', 0),
    ('last_pr', 1),
    ('commits', 0),
    ('files_changes', 0),
    ('src_churn', 0),
    ('src_addition', 0),
    ('src_deletion', 0),
    ('pr_file_merged_proportion', 0),
    ('pr_file_rejected_count', 0),
    ('text_similarity_merged', 0),
    ('pr_file_rejected_proportion', 0),
    ('has_test', 1),
    ('has_body', 1),
    ('pr_file_merged_count', 0),
    ('pr_file_submitted_count', 0),
    ('test_churn', 0),
    ('text_forward_link', 1),
    ('sloc', 0),
    ('recent_3_month_project_pr_num', 0),
    ('text_code_proportion', 0),
    ('team_size', 0),
    ('file_similarity_rejected', 0),
    ('text_similarity_rejected', 0),
    ('file_similarity_merged', 0),
    ('recent_project_passrate', 0),
    ('test_lines_per_kloc', 0),
    ('recent_3_month_commit', 0),
    ('perc_ext_contribs', 0),
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
def run_monthly(client, clf, print_prf=False, print_prf_each=False, print_main_proportion=False, print_AUC=False, MonthGAP=1, persistence=False, ayonel_numerical_attr=None, ayonel_boolean_attr=None):

    data_dict, pullinfo_list_dict = load_data_monthly(ayonel_numerical_attr=ayonel_numerical_attr, ayonel_boolean_attr=ayonel_boolean_attr,
                                  ayonel_categorical_attr_handler=ayonel_categorical_attr_handler, MonthGAP=MonthGAP)
    accuracy = 0
    AUC = 0
    mean_merged_precision, mean_merged_recall, mean_merged_F1, \
    mean_rejected_precision, mean_rejected_recall, mean_rejected_F1 = 0, 0, 0, 0, 0, 0

    for org, repo in org_list:
        # print(org+",", end='')
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

        for batch in batch_iter:
            if len(batch[0]) == 0:  # 测试集没有数据，直接预测下一batch
                continue
            test_X = np.array(batch[0])
            test_y = np.array(batch[1])
            # 正常训练
            train(clf, train_X, train_y)

            actual_result += test_y.tolist()  # 真实结果
            predict_result += clf.predict(test_X).tolist()  # 预测结果
            predict_result_prob += [x[0] for x in clf.predict_proba(test_X).tolist()]
            mean_accuracy += clf.score(test_X, test_y)
            train_X = np.concatenate((train_X, test_X))
            train_y = np.concatenate((train_y, test_y))

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

        accuracy += acc_num / len(actual_result)

        # precision, recall, F1 = precision_recall_f1(predict_result, actual_result)
        #
        # if print_prf:
        #     print(',%f,%f,%f' % (precision, recall, F1), end='')
        #
        if print_prf_each:
            try:
                merged_precision, merged_recall, merged_F1 = precision_recall_f1(predict_result, actual_result, POSITIVE=0)
                if mean_merged_precision == -1 or mean_merged_recall == -1 or mean_merged_F1 == -1:
                    pass
                else:
                    mean_merged_precision += merged_precision
                    mean_merged_recall += merged_recall
                    mean_merged_F1 += merged_F1
            except ValueError:
                mean_merged_precision, mean_merged_recall, mean_merged_F1 = -1, -1, -1


            try:
                rejected_precision, rejected_recall, rejected_F1 = precision_recall_f1(predict_result, actual_result, POSITIVE=1)
                if mean_rejected_precision == -1 or mean_rejected_recall == -1 or mean_rejected_F1 == -1:
                    pass
                else:
                    mean_rejected_precision += rejected_precision
                    mean_rejected_recall += rejected_recall
                    mean_rejected_F1 += rejected_F1
            except ValueError:
                mean_rejected_precision, mean_rejected_recall, mean_rejected_F1 = -1, -1, -1

            # print(',%f,%f,%f,%f,%f,%f' % (merged_F1, merged_precision, merged_recall, rejected_F1,rejected_precision, rejected_recall ), end='')

        if print_main_proportion:
            main_proportion = predict_result.count(1) / len(predict_result)
            print(',%f' % (main_proportion if main_proportion > 0.5 else 1 - main_proportion), end='')

        if print_AUC:
            y = np.array(actual_result)
            pred = np.array(predict_result_prob)
            fpr, tpr, thresholds = roc_curve(y, pred)
            this_AUC = auc(fpr, tpr)
            AUC += this_AUC if this_AUC > 0.5 else 1 - this_AUC
            # print(',%f' % (AUC if AUC > 0.5 else 1 - AUC), end='')
        # print()
    if mean_merged_precision == -1 or mean_merged_recall == -1 or mean_merged_F1 == -1 \
        or mean_rejected_precision == -1 or mean_rejected_recall == -1 or mean_rejected_F1 == -1:
        return accuracy/len(org_list), AUC/len(org_list), -1, -1, -1, -1, -1, -1
    else:
        return accuracy / len(org_list), AUC / len(org_list), \
               mean_merged_precision/len(org_list), \
               mean_merged_recall/len(org_list), \
               mean_merged_F1/len(org_list), \
               mean_rejected_precision/len(org_list), \
               mean_rejected_recall/len(org_list), \
               mean_rejected_recall/len(org_list)



if __name__ == '__main__':

    clf = XGBClassifier(seed=RANDOM_SEED)

    outfile = open('feature_selection/feature_selection.csv', "w", encoding="utf-8", newline="")
    writer = csv.writer(outfile)
    base = 0
    AUC = 0
    mean_merged_precision, mean_merged_recall, mean_merged_F1,\
    mean_rejected_precision, mean_rejected_recall, mean_rejected_F1,=0,0,0,0,0,0
    accuracy = 0
    best_ayonel_numerical_attr = []
    best_ayonel_boolean_attr = []
    print(str(base) + '---->' + str(AUC) + ', 数组还剩：' + str(len(attrs)))
    round = 1
    while True:
        best_attr = ''
        best_type = 0
        print("第"+str(round)+"轮开始：")
        for attr, type in attrs:
            ayonel_numerical_attr = best_ayonel_numerical_attr.copy()
            ayonel_boolean_attr = best_ayonel_boolean_attr.copy()
            if type == 0:  # 数值属性
                ayonel_numerical_attr.append(attr)
            else:  # bool属性
                ayonel_boolean_attr.append(attr)
            print('ayonel_numerical_attr:' + str(ayonel_numerical_attr))
            print('ayonel_boolean_attr:' + str(ayonel_boolean_attr))
            print('*****************************')
            this_accuracy, this_AUC,\
            this_mean_merged_precision, this_mean_merged_recall, this_mean_merged_F1, \
            this_mean_rejected_precision, this_mean_rejected_recall, this_mean_rejected_F1 = \
                run_monthly(clf, print_prf=False, print_prf_each=True,
                                print_main_proportion=False,
                                print_AUC=True, MonthGAP=6, persistence=False,
                                ayonel_numerical_attr=ayonel_numerical_attr,
                                ayonel_boolean_attr=ayonel_boolean_attr)

            if this_AUC > AUC:
                mean_merged_precision, mean_merged_recall, mean_merged_F1, \
                mean_rejected_precision, mean_rejected_recall, mean_rejected_F1 = \
                    this_mean_merged_precision, this_mean_merged_recall, this_mean_merged_F1, \
                    this_mean_rejected_precision, this_mean_rejected_recall, this_mean_rejected_F1
                AUC = this_AUC
                accuracy = this_accuracy
                best_attr = attr
                best_type = type

        if AUC <= base:  # AUC 没有提高
            break

        attrs.remove((best_attr, best_type))
        print(str(base) + '---->' + str(AUC)+', 数组还剩：'+str(len(attrs))+",本轮选择属性为："+best_attr)
        base = AUC
        if best_type == 0:  # 数值属性
            best_ayonel_numerical_attr.append(best_attr)
        else:
            best_ayonel_boolean_attr.append(best_attr)

        writer.writerow([best_attr,
                         mean_merged_precision, mean_merged_recall, mean_merged_F1,
                         mean_rejected_precision, mean_rejected_recall, mean_rejected_F1,
                         accuracy, AUC])
        outfile.flush()
        print("本轮最优结果，acc:" + str(accuracy) + ',AUC:' + str(AUC))
        round += 1

    print('ayonel_numerical_attr:' + str(best_ayonel_numerical_attr))
    print('ayonel_boolean_attr:' + str(best_ayonel_boolean_attr))
    outfile.close()

# 单特征分类器结果计算。
# if __name__ == '__main__':
#     clf = XGBClassifier(seed=RANDOM_SEED)
#     # clf = RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced_subsample')
#     # clf = CostSensitiveBaggingClassifier()
#
#
#     AUC = 0
#     accuracy = 0
#     best_attr = ''
#     best_type = 0
#     for attr, type in attrs:
#         ayonel_numerical_attr=[]
#         ayonel_boolean_attr=[]
#         if type == 0:
#             ayonel_numerical_attr.append(attr)
#         else:
#             ayonel_boolean_attr.append(attr)
#
#         accuracy, AUC = run_monthly(clf, print_prf=False, print_prf_each=False, print_main_proportion=False,
#                                               print_AUC=True, MonthGAP=6, persistence=False,
#                                               ayonel_numerical_attr=ayonel_numerical_attr,
#                                               ayonel_boolean_attr=ayonel_boolean_attr)
#
#         print(attr+','+str(accuracy)+','+str(AUC))