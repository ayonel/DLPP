'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
 统计文件后缀
'''
from src.constants import *
from src.database import dbutil
from src.constants import org_list
from collections import Counter

if __name__ == '__main__':
    client = dbutil.get_connection()

    for org, repo in org_list:
        pullinfo_list = list(client[org]['pullinfo'].find())
        merged_number = set()
        for pullinfo in pullinfo_list:
            if pullinfo['merged']:
                merged_number.add(pullinfo['number'])

        merged_file_dict = {}
        rejected_file_dict = {}

        pullfile_list = list(client[org]['pullfile'].find())

        for pullfile in pullfile_list:
            if int(pullfile['number']) in merged_number:
                if '.' in pullfile['filename']:
                    postfix = pullfile['filename'].split('.')[-1]
                else:
                    postfix = ''
                merged_file_dict[postfix] = merged_file_dict.get(postfix, 0) + 1
            else:
                if '.' in pullfile['filename']:
                    postfix = pullfile['filename'].split('.')[-1]
                else:
                    postfix = ''
                rejected_file_dict[postfix] = rejected_file_dict.get(postfix, 0) + 1
        merged_counter = Counter(merged_file_dict)
        rejected_counter = Counter(rejected_file_dict)
        print(org)
        print(merged_counter)
        print(rejected_counter)


