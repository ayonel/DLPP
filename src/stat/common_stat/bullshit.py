'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
'''

from src.database.dbutil import *
from src.constants import *
from collections import Counter
import csv
@mongo
def pullcommit(client):
    for org, repo in [('zendframework','zendframework'),
                      ('Katello', 'katello'),
                      ('Baystation12', 'Baystation12')]:

        pullcommits = client[org]['pullcommit'].find()
        pullcommit_dict = {}
        for pullcommit in pullcommits:
            if pullcommit['number'] not in pullcommit_dict:
                pullcommit_dict[pullcommit['number']] = 1
            else:
                pullcommit_dict[pullcommit['number']] += 1


        with open("data/"+org+'.csv', "w", newline='') as f:
            writer = csv.writer(f)
            for k, v in pullcommit_dict.items():
                writer.writerow([k, v])

        continue



        c_d = {}

        for k,v in pullcommit_dict.items():
            if v not in c_d:
                c_d[v] = 1
            else:
                c_d[v] += 1

        # print(c_d)
        count = 0
        for i in range(1, 152):
            if i in c_d:
                print(str(i)+','+str(c_d[i]))
                count += i * c_d[i]
            else:
                print(str(i)+','+str(0))
        print(count)
        exit()



@mongo
def commit(client):
    for org, repo in [('Baystation12','Baystation12')]:


        commitfiles = client[org]['commitfile'].find()
        d = {}
        for commitfile in commitfiles:
            if commitfile['commit_sha'] not in d:
                d[commitfile['commit_sha']] = 1
            else:
                d[commitfile['commit_sha']] += 1


        c_d = {}
        count = 0
        for k,v in d.items():
            if v not in c_d:
                c_d[v] = 1
            else:
                c_d[v] += 1

        for k,v in c_d.items():
            count += v*k
        print(count)

        for k,v in c_d.items():
            print(str(k)+','+str(v))





if __name__ == '__main__':
    pullcommit()