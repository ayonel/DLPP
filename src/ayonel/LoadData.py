# 用于数据加载

from src.constants import *
from src.database.dbutil import *


# 归一化数值属性，主要将数值属性规约到[0,1]区间
def regular(attrdata_list, ayonel_regular_attr):
    for arg in ayonel_regular_attr:
        max_v = max(attrdata_list, key=lambda x: float(x[arg]))[arg]
        min_v = min(attrdata_list, key=lambda x: float(x[arg]))[arg]
        max_gap = max_v - min_v
        for k, v in enumerate(attrdata_list):
            attrdata_list[k][arg] = (v[arg] - min_v) / max_gap


# 将bool类型转换为int
def bool2int(attrdata_list, ayonel_boolean_attr):
    for k, v in enumerate(attrdata_list):
        for attr in ayonel_boolean_attr:
            attrdata_list[k][attr] = 1 if v[attr] else 0


# 将week属性变成[1,0,0,0,0,0,0]列表
def week_handler(week):
    return week_feature_dict[week]


# 获取commits数量在50+的pr
def getBadSet(org, client):
    return set([x['number'] for x in list(client[org]['pullinfo'].find({'commits': {'$gt': 50}}, {'number':1}))])

# 定义一个按月份划分的数据集迭代器
class MonthData(object):
    def __init__(self, data, gap=1):
        self.X = data[0]
        self.y = data[1]
        self.gap = gap
        self.start_month = self.X[0][0]  # 第一个月
        self.end_month = self.start_month + self.gap
        self.cursor = 0
        self.length = len(data[0])

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor < self.length:
            X_batch = []
            y_batch = []
            count = 0
            for i in range(self.cursor, self.length):
                if self.X[i][0] >= self.start_month and self.X[i][0] < self.end_month:
                    X_batch.append(self.X[i][1:])
                    y_batch.append(self.y[i])
                    count += 1
                else:
                    break
            self.start_month = self.end_month
            self.end_month = self.start_month + self.gap
            self.cursor += count
            return (X_batch, y_batch)
        raise StopIteration()

    def reset(self):
        self.start_month = self.X[0][0]  # 第一个月
        self.end_month = self.start_month + self.gap
        self.cursor = 0


# 直接加载完整的数据集
@mongo
def load_data(client, ayonel_numerical_attr=None, ayonel_boolean_attr=None, ayonel_categorical_attr_handler=None):
    attr_dict = {}      # 键为每个项目，值为每个项目的X
    label_dict = {}     # 键为每个项目，值为每个项目的y
    pullinfo_list_dict = {}  # 键为每个项目，值为每个项目的pr信息列表
    for org, repo in org_list:
        # 获取属性信息
        attrdata_list = sorted(list(client[org]['ayonel'].find()), key=lambda x: int(x['number']))
        # 归一化
        # regular(attrinfo_list, ayonel_regular_attr)

        # bool转int
        bool2int(attrdata_list, ayonel_boolean_attr)
        for attr, handler in ayonel_categorical_attr_handler:
            for k, v in enumerate(attrdata_list):
                attrdata_list[k][attr] = handler(v[attr])

        attr_list = []
        # 先构造一个字典，key-value分别为number和merged
        is_merged = {}
        pullinfo_list = list(client[org]['pullinfo'].find({}).sort('number'))
        for pullinfo in pullinfo_list:
            is_merged[str(pullinfo['number'])] = pullinfo['merged']
        for pull in attrdata_list:
            L = [pull[attr] for attr in ayonel_numerical_attr + ayonel_boolean_attr]
            for attr, handler in ayonel_categorical_attr_handler:
                L += pull[attr]
            attr_list.append(L)
        attr_dict[org] = attr_list
        # 构造标签y
        label_list = []
        for pull in attrdata_list:
            label_list.append(0 if is_merged[pull['number']] else 1)
        label_dict[org] = label_list
        pullinfo_list_dict[org] = pullinfo_list

    # print('data loaded over!!!')
    return (attr_dict, label_dict, pullinfo_list_dict)

# 按月份划分加载数据集
@mongo
def load_data_monthly(client, ayonel_numerical_attr=None, ayonel_boolean_attr=None, ayonel_categorical_attr_handler=None, MonthGAP=1):
    data_dict = {}
    pullinfo_list_dict = {}  # 键为每个项目，值为每个项目的pr信息列表
    for org, repo in org_list:
        # 获取属性信息
        attrdata_list = sorted(list(client[org]['ayonel'].find()), key=lambda x: int(x['number']))
        # 归一化
        # regular(attrdata_list, ayonel_regular_attr)
        # bool2int
        bool2int(attrdata_list, ayonel_boolean_attr)
        # 处理categorical 属性
        for attr, handler in ayonel_categorical_attr_handler:
            for k, v in enumerate(attrdata_list):
                attrdata_list[k][attr] = handler(v[attr])
        # 构造一个字典，key-value分别为number和merged
        is_merged = {}
        # 构造一个字典，key-value分别为number和month
        month_dict = {}
        pullinfo_list = list(client[org]['pullinfo'].find({}).sort('number'))
        for pullinfo in pullinfo_list:
            is_merged[pullinfo['number']] = pullinfo['merged']
            month_dict[pullinfo['number']] = pullinfo['month']

        # 添加月份信息
        attr_list = []
        for pull in attrdata_list:
            L = [month_dict[int(pull['number'])]]+[pull[attr] for attr in ayonel_numerical_attr+ayonel_boolean_attr]
            for attr, handler in ayonel_categorical_attr_handler:
                L += pull[attr]
            attr_list.append(L)

        # 添加label信息
        label_list = []
        for pull in attrdata_list:
            label_list.append(0 if is_merged[int(pull['number'])] else 1)

        data_dict[org] = MonthData((attr_list, label_list), gap=MonthGAP)
        pullinfo_list_dict[org] = pullinfo_list
    return data_dict, pullinfo_list_dict



