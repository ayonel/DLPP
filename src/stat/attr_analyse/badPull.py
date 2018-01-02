'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
 
 查看每个项目中file数特别多的pr
'''
from src.database.dbutil import *
from src.constants import *
from src.utils import *
import nltk
import re


@mongo
def remove_self_close(client):
    for org, repo in org_list:
        all = client[org]['pullinfo'].find().count()
        bad = client[org]['pullinfo'].find({'commits':{'$gt': 50}}).count()

        print(repo+"    " + str(all) +"     "+str(bad)+"    "+str(bad/all))


if __name__ == '__main__':
    remove_self_close()
