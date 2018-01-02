# coding: utf-8

from src.constants import *
from src.database.dbutil import *


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
                if self.start_month <= self.X[i][0] < self.end_month:
                    X_batch.append(self.X[i][1:])
                    y_batch.append(self.y[i])
                    count += 1
                else:
                    break
            self.start_month = self.end_month
            self.end_month = self.start_month + self.gap
            self.cursor += count
            return X_batch, y_batch
        raise StopIteration()
    def reset(self):
        self.start_month = self.X[0][0]  # 第一个月
        self.end_month = self.start_month + self.gap
        self.cursor = 0


@mongo
def load_data(client, gousios_attr_list=None):
    attr_dict = {}      # 键为每个项目，值为每个项目的X
    label_dict = {}     # 键为每个项目，值为每个项目的y

    for org, repo in org_list:
        # 获取属性信息
        attrdata_list = sorted(list(client[org]['gousios'].find({'is_irec': True})), key=lambda x: int(x['number']))
        attr_list = []

        for pull in attrdata_list:
            attr_list.append([pull[attr] for attr in gousios_attr_list])
        attr_dict[org] = attr_list

        # 先构造一个字典，key-value分别为number和merged
        is_merged = {}
        pullinfo_list = list(client[org]['pullinfo'].find({},{'number': 1, 'merged': 1, '_id': 0}))
        for pullinfo in pullinfo_list:
            is_merged[str(pullinfo['number'])] = pullinfo['merged']
        label_list = []
        for pull in attrdata_list:
            label_list.append(0 if is_merged[pull['number']] else 1)
        label_dict[org] = label_list
    return attr_dict, label_dict



@mongo
def load_data_monthly(client, gousios_attr_list=None, MonthGAP=None):
    data_dict = {}
    for org, repo in org_list:
        # 获取属性信息
        attrdata_list = sorted(list(client[org]['gousios'].find({'is_irec': True})), key=lambda x: int(x['number']))
        # 构造标签y
        # 先构造一个字典，key-value分别为number和merged
        is_merged = {}
        month_dict = {}
        pullinfo_list = list(client[org]['pullinfo'].find({}, {'number': 1, 'merged': 1, '_id': 0, 'month': 1}).sort(('number')))
        for pullinfo in pullinfo_list:
            is_merged[pullinfo['number']] = pullinfo['merged']
            month_dict[pullinfo['number']] = pullinfo['month']

        attr_list = []
        for pull in attrdata_list:
            pull['test'] = 0
            attr_list.append([month_dict[int(pull['number'])]]+[pull[attr] for attr in gousios_attr_list])
        # 添加label信息
        label_list = []
        for pull in attrdata_list:
            label_list.append(0 if is_merged[int(pull['number'])] else 1)

        data_dict[org] = MonthData((attr_list, label_list), gap=MonthGAP)
    return data_dict
if __name__ == '__main__':
    load_data_monthly()
