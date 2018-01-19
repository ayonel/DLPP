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
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import train_test_split as tts
from imblearn.combine import SMOTEENN
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
from costcla.models import CostSensitiveRandomForestClassifier
from costcla.models import CostSensitiveBaggingClassifier

SEG_PROPORTION = 8/10
FOLDS = 5
# 按重要性排序之后的

from src.ayonel.clf import ayonel_boolean_attr, ayonel_numerical_attr, ayonel_categorical_attr_handler

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
def run_monthly(client, MonthGAP=1):
    data_dict, pullinfo_list_dict = load_data_monthly(ayonel_numerical_attr=ayonel_numerical_attr, ayonel_boolean_attr=ayonel_boolean_attr,
                                  ayonel_categorical_attr_handler=ayonel_categorical_attr_handler, MonthGAP=MonthGAP)

    for org, repo in [('Baystation12','xx')]:
        print(org+",")
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
            parameters = [
                ("learning_rate", iandfrange(0, 1, 0.05)),
                ("gamma", iandfrange(0, 1, 0.1) + iandfrange(1, 10, 1)),
                ("objective", ["binary:logistic", "rank:pairwise"]),
                ("max_depth", range(1, len(batch[0][0]) + 1)),
                ("min_child_weight", range(1, len(batch[0][0]) + 1)),
                ("subsample", iandfrange(0, 1.01, 0.1)),
                ("colsample_bytree", iandfrange(0.1, 1.01, 0.1)),
                # ("scale_pos_weight",[np.sum(test_y == 0) / np.sum(test_y == 1)])
            ]

            tuned_params = {}  # 已调好的参数
            for k, v in enumerate(parameters):
                tuning_param = {}
                tuning_param[v[0]] = v[1]
                estimator_xg = XGBClassifier(seed=RANDOM_SEED, **tuned_params)
                clf = GridSearchCV(
                    estimator=estimator_xg,
                    param_grid=tuning_param,
                    scoring="roc_auc", cv=5)
                clf.fit(train_X, train_y)
                tuned_params = dict(tuned_params, **clf.best_params_)
            print(tuned_params)
            # 入库
            client[org]['model'].update({'round': round, 'model': 'xgboost', 'gap': MonthGAP}, {'$set': tuned_params}, upsert=True)

            best_est = XGBClassifier(seed=RANDOM_SEED, **tuned_params)

            train(best_est, train_X, train_y)

            print(best_est.score(test_X, test_y))

            actual_result += test_y.tolist()  # 真实结果
            predict_result += best_est.predict(test_X).tolist()  # 预测结果
            predict_result_prob += [x[0] for x in best_est.predict_proba(test_X).tolist()]
            mean_accuracy += best_est.score(test_X, test_y)
            train_X = np.concatenate((train_X, test_X))
            train_y = np.concatenate((train_y, test_y))
            round += 1

        acc_num = 0
        for i in range(len(actual_result)):
            if actual_result[i] == predict_result[i]:
                acc_num += 1
        print(acc_num / len(actual_result))

if __name__ == '__main__':
    run_monthly(MonthGAP=6)


