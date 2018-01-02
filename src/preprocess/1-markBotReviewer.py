# 通过对30项目的reviewer进行搜索，查询是否有*bot*的人出现
# 从YYRec中提取每个项目的reviewer，并存入数据库

from src.constants import org_list
from src.database.dbutil import *
import re


@mongo
def main(client):
    for org, repo in org_list:
        print('-----'+org+'-----')
        with open('D:/LabCode/ReviewerRec/YYRec/repo/' + org + '/reviewer.txt', 'r') as f:
            reviewer_list = [x.strip('\r\n') for x in f.readlines()]
            insert_data = []
            for reviewer in reviewer_list:
                if re.match(r'.*bot.*', reviewer):
                    print(org + ':' + reviewer)
                insert_data.append({'name': reviewer})
            client[org]['reviewer'].insert(insert_data)

if __name__ == '__main__':
    main()


'''
Baystation12:bs12-bot
dlang:klickverbot
joomla:joomla-cms-bot
joomla:jissues-bot
rails:rails-bot
'''
