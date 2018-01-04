'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
'''

#精准率
def precision_recall_f1(predict_result, actual_result, POSITIVE=None):
    if len(predict_result) != len(actual_result):
        raise ValueError("预测结果集与真实结果集大小不一致")

    '''
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*TP(2*TP+FP+FN)
    '''
    if not POSITIVE:
        POSITIVE = 1 if actual_result.count(1)/len(actual_result) > 0.5 else 0
    TP, FP, FN = 0, 0, 0
    for i in range(len(predict_result)):
        if predict_result[i] == POSITIVE:
            if actual_result[i] == POSITIVE:
                TP += 1
            else:
                FP += 1
        elif actual_result[i] == POSITIVE:
            FN += 1

    if TP+FP == 0:
        raise ValueError("预测正类数为0")

    if TP + FN == 0:
        raise ValueError("真是正类数为0")

    return TP/(TP + FP), TP / (TP + FN), 2*TP/(2*TP+FP+FN)


