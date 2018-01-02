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

attr_dict, label_dict = load_data()
def train(clf, X, y):
    clf.fit(X, y)


def blending(rf, xg, ad, test_X, test_y):
    best_accuracy = 0.0
    best_rf_weight = 0.0
    best_xg_weight = 0.0
    best_ad_weight = 0.0

    for i in iandfrange(0, 1.01, 0.1):
        for j in iandfrange(0, 1.01-i, 0.1):
            proba = np.array(rf.predict_proba(test_X)) * i
            proba += np.array(xg.predict_proba(test_X)) * j
            proba += np.array(ad.predict_proba(test_X)) * (1-i-j)
            predict_result = np.array([0 if x[0] >= x[1] else 1 for x in proba])

            hit = 0
            for k in range(len(test_X)):
                if int(test_y[k]) == predict_result[k]:
                    hit += 1
            accuracy = hit / len(test_X)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_rf_weight = i
                best_xg_weight = j
                best_ad_weight = 1-i-j
    return (best_accuracy, best_rf_weight, best_xg_weight, best_ad_weight)



if __name__ == '__main__':
    client = get_connection()
    csv_writer = getCSVWriter(ROOT_PATH + "/src/stat/result_stat/"+getTodayString()+".csv", 'a')
    for org, repo in [('dlang', 'dmd')]:
        print(org)
        one_line = [org, repo]

        input_X = attr_dict[org]
        input_y = label_dict[org]
        seg_point = int(len(attr_dict[org])*SEG_PROPORTION)

        train_X = np.array(attr_dict[org][:seg_point])
        train_y = np.array(label_dict[org][:seg_point])

        test_X = np.array(attr_dict[org][seg_point:])
        test_y = np.array(label_dict[org][seg_point:])

        estimators = []
        ########################################随机森林##################################
        parameters = client[org]['model'].find_one({'model': 'RandomForestClassifier'}, {'_id': 0, 'model': 0})
        estimator_rf = RandomForestClassifier(random_state=RANDOM_SEED, **parameters)
        train(estimator_rf, train_X, train_y)
        print(estimator_rf.score(test_X, test_y))
        one_line.append(estimator_rf.score(test_X, test_y))
        #########################################xgboost##################################
        parameters = client[org]['model'].find_one({'model': 'XGBClassifier'}, {'_id': 0, 'model': 0})
        estimator_xg = XGBClassifier(seed=RANDOM_SEED, **parameters)
        train(estimator_xg, train_X, train_y)
        print(estimator_xg.score(test_X, test_y))

        one_line.append(estimator_xg.score(test_X, test_y))
        ##########################################Adboost##################################
        parameters = client[org]['model'].find_one({'model': 'AdaBoostClassifier'}, {'_id': 0, 'model': 0})
        estimator_ad = AdaBoostClassifier(random_state=RANDOM_SEED, **parameters)
        train(estimator_ad, train_X, train_y)
        print(estimator_ad.score(test_X, test_y))
        exit()
        one_line.append(estimator_ad.score(test_X, test_y))
        ###############################################################################
        blending_result = (blending(estimator_rf, estimator_xg, estimator_ad, test_X=test_X, test_y=test_y))
        for item in blending_result:
            one_line.append(item)
        csv_writer.writerow(one_line)

        #
        # accuracy = clf.score(test_X, test_y)
        # client[org]['model'].update({'model': clf.estimator.__class__.__name__}, {'$set': clf.best_params_}, upsert=True)
        # print(repo)
        # print(clf.best_params_)
        
        # # client[org]['result'].insert({clf.__class__.__name__: accuracy})
        # print(accuracy)






