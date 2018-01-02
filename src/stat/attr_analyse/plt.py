'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
'''
from src.database.dbutil import *
import matplotlib.pyplot as plt
import numpy as np

client = get_connection()
attrdata_list = list(client['Baystation12']['ayonel'].find())
pullinfo_list = list(client['Baystation12']['pullinfo'].find())

is_merged = {}
for pullinfo in pullinfo_list:
    is_merged[pullinfo['number']] = pullinfo['merged']
axes = plt.subplot(111)


merged_samples_X = [vec['commits'] for vec in attrdata_list if is_merged[int(vec['number'])]]
rejected_samples_X = [vec['commits'] for vec in attrdata_list if not is_merged[int(vec['number'])]]


type1 = axes.scatter(merged_samples_X, [1]*len(merged_samples_X), s=20, c='red')
type2 = axes.scatter(rejected_samples_X, [2]*len(rejected_samples_X), s=20, c='green')

axes.legend((type1, type2), ('merge', 'reject'), loc=2)
plt.show()