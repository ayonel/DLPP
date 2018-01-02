import numpy as np
import re
from src.constants import *
from src.database.dbutil import *


MAX_WORD_COUNT = '95'  # 以数据集中95%的评论长度为输入

# 加载评论信息，构造输入
def load_comment_and_lables(org):
    client = get_connection()
    # 获取reviewer集合
    if org in bot_reviewer:
        reviewer_set = set([x['name'] for x in list(client[org]['reviewer'].find())]) - set(bot_reviewer[org])
    else:
        reviewer_set = set([x['name'] for x in list(client[org]['reviewer'].find())])

    # 对每个pr构造是否接受标签
    number_list = []
    pull_dict = {}
    pullinfo_list = list(client[org]['pullinfo'].find({'created_at': {'$lt': STMAP_2016_7_31}}).sort('number', pymongo.ASCENDING))
    for pullinfo in pullinfo_list:
        number_list.append(pullinfo['number'])
        pull_dict[pullinfo['number']] = {}
        pull_dict[pullinfo['number']]['comment'] = ''
        pull_dict[pullinfo['number']]['info'] = pullinfo['title_token']+ " " + pullinfo['body_token']
        pull_dict[pullinfo['number']]['label'] = [1, 0] if pullinfo['merged'] else [0, 1]

    # 合并comment, 只算属于reviewer的issuecomment
    pullcomment_list = list(client[org]['pullcomment'].find({'inline': 0, 'created_at': {'$lt': STMAP_2016_7_31},'commentor':{'$in': list(reviewer_set)}})
                            .sort([('number', pymongo.ASCENDING), ('created_at', pymongo.ASCENDING)]))

    for pullcomment in pullcomment_list:
        pull_dict[int(pullcomment['number'])]['comment'] += pullcomment['body_token']


    # 对评论按照词数分布，进行过滤，默认为95%
    max_word_count = int(client[org]['stat'].find_one()['pullinfo_word_distribution'][MAX_WORD_COUNT])
    print(org+'最大词数:'+str(max_word_count))
    for pull in pull_dict:
        # comment_word_count = pull_dict[pull]['comment'].count(' ')
        info_word_count = pull_dict[pull]['info'].count(' ')
        if info_word_count > max_word_count:
            pull_dict[pull]['info'] = ' '.join(pull_dict[pull]['info'].split(' ', max_word_count)[:max_word_count])
            # pull_dict[pull]['comment'] = ''
    #     if comment_word_count + info_word_count > max_word_count:
    #         pull_dict[pull]['comment'] = pull_dict[pull]['comment'].split(' ', comment_word_count+1-max_word_count+info_word_count)[-1]

    # x_text = [pull_dict[number]['info'] + pull_dict[number]['comment'] for number in number_list]
    x_text = [pull_dict[number]['info'] for number in number_list]
    y_label = np.array([pull_dict[number]['label'] for number in number_list])
    x_info = [pull_dict[number]['info'] for number in number_list]
    return [x_info, x_text, y_label]



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    # batch_iter = batch_iter([1,2,3,4,5,6,7,8], 2, 5, True)
    # for batch in batch_iter:
    #     print(batch)
    pass

