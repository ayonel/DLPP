'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
'''
import csv
import numpy as np
from collections import Counter
from src.gousios.LoadData import *
from src.constants import *
from src.eval.eval_utils import precision_recall_f1
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN


SEG_PROPORTION = 8/10
FOLDS = 5
gousios_attr_list = [
    'num_commits',
    'src_churn',
    'test_churn',
    'files_changes',
    'sloc',
    'team_size',
    'perc_ext_contribs',
    'commits_files_touched',
    'test_lines_per_kloc',
    'prev_pullreqs',
    'requester_succ_rate'
]


def train(clf, X, y):
    clf.fit(X, y)


def run(clf, print_prf=False, print_main_proportion=False):
    attr_dict, label_dict = load_data(gousios_attr_list=gousios_attr_list)
    ACC = 0.0
    for org, repo in org_list:
        input_X = attr_dict[org]
        input_y = label_dict[org]
        # print(sorted(Counter(input_y).items()))

        seg_point = int(len(input_X) * SEG_PROPORTION)

        train_X = np.array(input_X[:seg_point])
        train_y = np.array(input_y[:seg_point])

        train_X, train_y = SMOTE().fit_sample(train_X, train_y)

        test_X = np.array(input_X[seg_point:])
        test_y = np.array(input_y[seg_point:])

        train(clf, train_X, train_y)
        actual_result = test_y.tolist()
        predict_result = clf.predict(test_X).tolist()

        accuracy = clf.score(test_X, test_y)

        ACC += accuracy

        print(accuracy, end='')
        precision, recall, F1 = precision_recall_f1(predict_result, actual_result)
        if print_prf:
            print(",%f,%f,%f" % (precision, recall, F1), end='')

        if print_main_proportion:
            main_proportion = predict_result.count(1) / len(predict_result)
            print(',%f' % (main_proportion if main_proportion > 0.5 else 1 - main_proportion), end='')
        print()
    print(str(ACC/len(org_list)))




def run_monthly(clf, print_prf=False, print_prf_each=False, print_main_proportion=False, print_AUC=False, MonthGAP=1):
    data_dict = load_data_monthly(gousios_attr_list=gousios_attr_list, MonthGAP=MonthGAP)
    for org,repo in org_list:
        print(org+",", end='')
        batch_iter = data_dict[org]
        train_batch = batch_iter.__next__()
        train_X = np.array(train_batch[0])
        train_y = np.array(train_batch[1])
        predict_result = []
        actual_result = []
        predict_result_prob = []
        samples = 0
        for batch in batch_iter:
            if len(batch[0]) == 0:  # 测试集没有数据，直接预测下一batch
                continue
            test_X = np.array(batch[0])
            test_y = np.array(batch[1])
            train(clf, train_X, train_y)


            actual_result += test_y.tolist()  # 真实结果
            predict_result += clf.predict(test_X).tolist()  # 预测结果
            predict_result_prob += [x[0] for x in clf.predict_proba(test_X).tolist()]
            samples += test_X.size


            train_X = np.concatenate((train_X, test_X))
            train_y = np.concatenate((train_y, test_y))

        acc_num = 0
        for i in range(len(actual_result)):
            if actual_result[i] == predict_result[i]:
                acc_num += 1
        print(acc_num/len(actual_result), end='')
        precision, recall, F1 = precision_recall_f1(predict_result, actual_result)
        if print_prf:
            print(",%f,%f,%f" % (precision, recall, F1), end='')

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
            fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
            AUC = auc(fpr, tpr)
            print(',%f' % (AUC if AUC > 0.5 else 1-AUC), end='')
        print()
if __name__ == '__main__':
    clf = RandomForestClassifier(random_state=RANDOM_SEED)
    run_monthly(clf, print_prf=False, print_prf_each=True, print_main_proportion=False, print_AUC=True, MonthGAP=6)
    # run(clf)


