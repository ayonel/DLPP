'''
 Author: ayonel
 Date:
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
 新增文本相似度以及文件相似度属性的计算
'''


from src.constants import *
from src.database.dbutil import *
from src.utils import memory_available
import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import SnowballStemmer
import numpy

LESS_WORD_THRESHOLD = 3

tfidf_vectorizer = TfidfVectorizer()
Stemmer = SnowballStemmer("english")

with open(ROOT_PATH+"/src/stopwords", "r") as f:
    stopword_list = [x.strip('\r\n') for x in f.readlines()]

STOP_WORDS = set(stopword_list)
GAP_SECOND = SECOND_3_MONTH
text_word_count_dict = {}
text_less_words = set()
title_word_count_dict = {}
title_less_words = set()
body_word_count_dict = {}
body_less_words = set()




def build_empty_result_dict(pull_list):
    result_dict= {}
    for number in pull_list:
        number = str(number)
        result_dict[number] = {}
        result_dict[number]['text_similarity_merged'] = 0.0     # ok
        result_dict[number]['body_similarity_merged'] = 0.0     # ok
        result_dict[number]['title_similarity_merged'] = 0.0     # ok
        result_dict[number]['text_similarity_rejected'] = 0.0  # ok
        result_dict[number]['title_similarity_rejected'] = 0.0
        result_dict[number]['body_similarity_rejected'] = 0.0
        result_dict[number]['file_similarity_merged'] = 0.0
        result_dict[number]['file_similarity_rejected'] = 0.0

    return result_dict


def precess(pull_list, pull_dict):
    '''
    预处理，将分好的词提词干，去停用词
    :param pull_list:
    :return:
    '''
    for pull in pull_list:
        title = pull_dict[pull]['title_token'].split(" ")
        body = pull_dict[pull]['body_token'].split(" ")
        # 提词干
        for k, v in enumerate(title):
            title[k] = Stemmer.stem(v)
        for k, v in enumerate(body):
            body[k] = Stemmer.stem(v)

        #  过滤停用词
        title = list(filter(lambda x: x not in STOP_WORDS, title))
        body = list(filter(lambda x: x not in STOP_WORDS, body))

        for word in title:
            if word not in title_word_count_dict:
                title_word_count_dict[word] = 1
            else:
                title_word_count_dict[word] += 1

        for word in body:
            if word not in body_word_count_dict:
                body_word_count_dict[word] = 1
            else:
                body_word_count_dict[word] += 1

        for word in body+title:
            if word not in text_word_count_dict:
                text_word_count_dict[word] = 1
            else:
                text_word_count_dict[word] += 1

        pull_dict[pull]['title'] = title
        pull_dict[pull]['body'] = body
        pull_dict[pull]['text'] = body+title

    # 构造小词频的集合
    for word in title_word_count_dict:
        if title_word_count_dict[word] < LESS_WORD_THRESHOLD:
            title_less_words.add(word)

    for word in body_word_count_dict:
        if body_word_count_dict[word] < LESS_WORD_THRESHOLD:
            body_less_words.add(word)

    for word in text_word_count_dict:
        if text_word_count_dict[word] < LESS_WORD_THRESHOLD:
            text_less_words.add(word)
    # 开始过滤词频小于3的词
    for pull in pull_list:
        pull_dict[pull]['title'] = list(filter(lambda x: x not in title_less_words, pull_dict[pull]['title']))
    for pull in pull_list:
        pull_dict[pull]['body'] = list(filter(lambda x: x not in body_less_words, pull_dict[pull]['body']))
    for pull in pull_list:
        pull_dict[pull]['text'] = list(filter(lambda x: x not in text_less_words, pull_dict[pull]['text']))


    # 开始计算 tfidf
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    title_corpus = []
    body_corpus = []
    text_corpus = []
    for pull in pull_list:
        title_corpus.append(" ".join(pull_dict[pull]['title']))
        body_corpus.append(" ".join(pull_dict[pull]['body']))
        text_corpus.append(" ".join(pull_dict[pull]['text']))
    title_tfidf = transformer.fit_transform(vectorizer.fit_transform(title_corpus)).toarray()
    body_tfidf = transformer.fit_transform(vectorizer.fit_transform(body_corpus)).toarray()
    text_tfidf = transformer.fit_transform(vectorizer.fit_transform(text_corpus)).toarray()

    for i in range(len(pull_list)):
        pull_dict[pull_list[i]]['title_tfidf'] = title_tfidf[i]
        pull_dict[pull_list[i]]['body_tfidf'] = body_tfidf[i]
        pull_dict[pull_list[i]]['text_tfidf'] = text_tfidf[i]


def cal_similarity(pull_list, pull_dict, result_dict):
    for k, pull in enumerate(pull_list):
        print(org+':'+str(pull))
        nowtime = pull_dict[pull]['created_at']

        merged_title_list, rejected_title_list = [], []
        merged_body_list, rejected_body_list = [], []
        merged_text_list, rejected_text_list = [], []

        for i in list(range(k))[::-1]:
            if pull_dict[pull_list[i]]['created_at'] < nowtime-GAP_SECOND:
                break
            else:
                if pull_dict[pull_list[i]]['merged'] and pull_dict[pull_list[i]]['merged_at'] < nowtime:
                    merged_title_list.append(pull_dict[pull_list[i]]['title_tfidf'])
                    merged_body_list.append(pull_dict[pull_list[i]]['body_tfidf'])
                    merged_text_list.append(pull_dict[pull_list[i]]['text_tfidf'])
                if not pull_dict[pull_list[i]]['merged'] and pull_dict[pull_list[i]]['closed_at'] < nowtime:
                    rejected_title_list.append(pull_dict[pull_list[i]]['title_tfidf'])
                    rejected_body_list.append(pull_dict[pull_list[i]]['body_tfidf'])
                    rejected_text_list.append(pull_dict[pull_list[i]]['text_tfidf'])
        if merged_title_list:
            result_dict[pull]['title_similarity_merged'] = numpy.average(cosine_similarity([pull_dict[pull]['title_tfidf']], merged_title_list))
        if merged_body_list:
            result_dict[pull]['body_similarity_merged'] = numpy.average(cosine_similarity([pull_dict[pull]['body_tfidf']], merged_body_list))
        if merged_text_list:
            result_dict[pull]['text_similarity_merged'] = numpy.average(cosine_similarity([pull_dict[pull]['text_tfidf']], merged_text_list))
        if rejected_title_list:
            result_dict[pull]['title_similarity_rejected'] = numpy.average(cosine_similarity([pull_dict[pull]['title_tfidf']], rejected_title_list))
        if rejected_body_list:
            result_dict[pull]['body_similarity_rejected'] = numpy.average(cosine_similarity([pull_dict[pull]['body_tfidf']], rejected_body_list))
        if rejected_text_list:
            result_dict[pull]['text_similarity_rejected'] = numpy.average(cosine_similarity([pull_dict[pull]['text_tfidf']], rejected_text_list))
    return result_dict

def sort_two_string(str1, str2):
    return max(str1, str2) + ',' + min(str1, str2)

def file_sim_1weight(str1, str2):
    folder1 = str1.count('/')
    folder2 = str2.count('/')
    deep = max(folder1, folder2) + 1
    if folder1 < folder2:
        str1, str2 = str2, str1
    piece1 = str1.split('/')
    piece2 = str2.split('/')
    score = 0.0
    for k, v in enumerate(piece1):
        if k >= len(piece2):
            break
        if v == piece2[k]:
            score += 1
        else:
            break
    return score / deep

def history_file_regular_1weight(pull_list, pull_dict, result_dict, file_sim_score_dict):
    '''
    :param sorted_number_list:
    :param pull_dict:
    :param result_dict:
    :param file_integrate_dict:
    :return:
    a/b/c/d.js 与a/b/e.js 的分为 (1+1)/1+1+1+1 = 1/2
    '''
    for k, pr in enumerate(pull_list):
        print(org+':'+str(pr)+':file')
        nowtime = pull_dict[pr]['created_at']
        this_file_list = pull_dict[pr]['prfile']
        if memory_available() < 0.1:
            file_sim_score_dict = {}
        merged_count, rejected_count = 0, 0
        merged_file_score, rejected_file_score = 0.0, 0.0
        for history_pr in pull_list[k-1::-1]:
            if pull_dict[history_pr]['created_at'] < nowtime-GAP_SECOND:
                break
            else:
                if pull_dict[history_pr]['merged'] and pull_dict[history_pr]['merged_at'] < nowtime:
                    merged_count += 1
                    history_file_list = pull_dict[history_pr]['prfile']
                    if len(this_file_list) != 0 and len(history_file_list) != 0:
                        score = 0
                        for this_file in this_file_list:
                            for history_file in history_file_list:
                                pair = sort_two_string(this_file, history_file)
                                if pair not in file_sim_score_dict:
                                    file_sim_score_dict[pair] = file_sim_1weight(this_file, history_file)
                                    score += file_sim_score_dict[pair]
                                else:
                                    score += file_sim_score_dict[pair]
                        merged_file_score += score / (len(this_file_list) * len(history_file_list))

                if not pull_dict[history_pr]['merged'] and pull_dict[history_pr]['closed_at'] < nowtime:
                    rejected_count += 1
                    history_file_list = pull_dict[history_pr]['prfile']
                    if len(this_file_list) != 0 and len(history_file_list) != 0:
                        score = 0
                        for this_file in this_file_list:
                            for history_file in history_file_list:
                                pair = sort_two_string(this_file, history_file)
                                if pair not in file_sim_score_dict:
                                    file_sim_score_dict[pair] = file_sim_1weight(this_file, history_file)
                                    score += file_sim_score_dict[pair]
                                else:
                                    score += file_sim_score_dict[pair]
                        rejected_file_score += score / (len(this_file_list) * len(history_file_list))

        if merged_count == 0:
            result_dict[pr]['file_similarity_merged'] = 0.0
        else:
            result_dict[pr]['file_similarity_merged'] = merged_file_score/merged_count
        if rejected_count == 0:
            result_dict[pr]['file_similarity_rejected'] = 0.0
        else:
            result_dict[pr]['file_similarity_rejected'] = rejected_file_score/rejected_count
    return result_dict

if __name__ == '__main__':
    client = get_connection()
    for org, repo in org_list:
        # 获取pull_list, pull_dict
        pullinfo_list = list(client[org]['pullinfo'].find().sort('number', pymongo.ASCENDING))
        pull_list = []
        pull_dict = {}
        file_sim_score_dict = {}
        for pullinfo in pullinfo_list:
            pull_list.append(str(pullinfo['number']))
            pull_dict[str(pullinfo['number'])] = pullinfo
        reviewer_set = set([x['name'] for x in list(client[org]['reviewer'].find())])

        # 为pull_dict添加prfile字段
        for pull in pull_dict:
            pull_dict[pull]['prfile'] = []
        for pullfile in list(client[org]['pullfile'].find()):
            if pullfile['number'] in pull_dict:
                pull_dict[pullfile['number']]['prfile'].append(pullfile['filename'])

        result_dict = build_empty_result_dict(pull_list)
        # 预处理
        precess(pull_list, pull_dict)

        result_dict = cal_similarity(pull_list, pull_dict, result_dict)
        result_dict = history_file_regular_1weight(pull_list, pull_dict, result_dict, file_sim_score_dict)

        for pr in result_dict:
            client[org]['ayonel'].update({'number': pr}, {'$set': result_dict[pr]})



