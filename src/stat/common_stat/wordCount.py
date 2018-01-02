'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
 统计拒绝pr 与 接受pr的title中出现次数最多的词
'''


from src.constants import *
from src.database import dbutil
from src.constants import org_list
from collections import Counter

with open(ROOT_PATH+"/src/stopwords", "r") as f:
    stopword_list = [x.strip('\r\n') for x in f.readlines()]

STOP_WORDS = set(stopword_list)

if __name__ == '__main__':
    client = dbutil.get_connection()
    for org, repo in org_list:
        merged_word_dict = {}
        rejected_word_dict = {}

        pullinfo_list = list(client[org]['pullinfo'].find())
        for pullinfo in pullinfo_list:
            if pullinfo['merged']:
                for word in pullinfo['title_token'].split(" "):
                    if word not in STOP_WORDS:
                        merged_word_dict[word] = merged_word_dict.get(word, 0) + 1
            else:
                for word in pullinfo['title_token'].split(" "):
                    if word not in STOP_WORDS:
                        rejected_word_dict[word] = rejected_word_dict.get(word, 0) + 1

        merged_word_Counter = Counter(merged_word_dict)
        rejected_word_Counter = Counter(rejected_word_dict)

        print(org)
        print(merged_word_Counter.most_common(50))
        # print(rejected_word_Counter.most_common(50))




