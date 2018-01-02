# 将pullcomment的body进行分词，并存入数据库。约定字段为body_token
# 将pullinfo的body进行分词，并存入数据库。约定字段为body_token
# 将pullinfo的title进行分词，并存入数据库。约定字段为title_token

from nltk.tokenize import RegexpTokenizer
from src.database.dbutil import *
from src.constants import org_list
import re

tokenizer = RegexpTokenizer(r'\w+')
#正则分词
def myTokenize(text):
    return tokenizer.tokenize(text)

def lowercase(word):
    return word.lower()


def get_token_string(o_string):
    # 过滤掉代码信息   ```<code>``` 以及 `<code>`
    if o_string and isinstance(o_string, str):
        string = re.sub(r'`.*?`', ' ', re.sub('[\r\n]', ' ', o_string))
        string = re.sub(r'`.*?`', ' ', re.sub('[\r\n]', ' ', string))
        return ' '.join([token for token in myTokenize(string)])
    else:
        return ''

@mongo
def main(client):
    for org, repo in org_list:
        print(org)
        pullcomment_list = list(client[org]['pullcomment'].find())
        for pullcomment in pullcomment_list:
            client[org]['pullcomment'].update({'_id': pullcomment['_id']}, {'$set': {'body_token': get_token_string(pullcomment['body'])}})

        pullinfo_list = list(client[org]['pullinfo'].find())
        for pullinfo in pullinfo_list:
            client[org]['pullinfo'].update({'_id': pullinfo['_id']}, {'$set': {'body_token': get_token_string(pullinfo['body'])}})
            client[org]['pullinfo'].update({'_id': pullinfo['_id']}, {'$set': {'title_token': get_token_string(pullinfo['title'])}})

if __name__ == '__main__':
    main()
