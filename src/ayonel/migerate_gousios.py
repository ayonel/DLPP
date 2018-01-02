'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
 description:将gousios的commits_files_touched，perc_ext_contribs迁移到ayonel
'''

from src.database.dbutil import *
from src.constants import *

@mongo
def migerate(client):
    for org, repo in org_list:
        print(org)
        gousios_attr = list(client[org]['gousios'].find({}, {'_id': 0, 'commits_files_touched': 1, 'perc_ext_contribs': 1, 'number': 1, 'team_size': 1}))
        for attr in gousios_attr:
            data = {
                'commits_files_touched': attr['commits_files_touched'],
                'perc_ext_contribs': attr['perc_ext_contribs'],
                'team_size': attr['team_size']
            }

            client[org]['ayonel'].update({'number': attr['number']}, {'$set': data}, upsert=True)

if __name__ == '__main__':
    for org,repo in org_list:
        print("mongodump -h 192.168.1.102 -d "+org)
