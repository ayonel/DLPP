# 为每个项目建立一个数据库,并且将其信息存入
# 并且删除2016年7月31号之后的数据
import pymongo
import src.database.dbutil as dbutil
import src.constants as constants
import src.database.CONFIG as CONFIG

if __name__ == '__main__':
    client = dbutil.get_connection()
    db = client[CONFIG.MONGODB_DBNAME]
    for org, repo in constants.org_list:
        print(org)
        pull_set = set()
        pull_list = list((db['pullinfo'].find({'org': org, 'created_at': {'$lt': constants.STMAP_2016_7_31}})))
        this_db = client[org]
        # 插入pullinfo
        for pull in pull_list:
            this_db['pullinfo'].insert(pull)
            pull_set.add(pull['number'])
        this_db['pullinfo'].create_index([('org', pymongo.ASCENDING),
                                          ('author', pymongo.ASCENDING),
                                          ('created_at', pymongo.ASCENDING),
                                          ('merge_commit_sha', pymongo.ASCENDING)])
        this_db['pullinfo'].create_index('number', unique=True)

        # 插入pullfile
        pullfile_list = list((db['pullfile'].find({'org': org})))
        for pullfile in pullfile_list:
            if int(pullfile['number']) in pull_set:
                this_db['pullfile'].insert(pullfile)
        this_db['pullfile'].create_index([('org', pymongo.ASCENDING),
                                          ('number', pymongo.ASCENDING)])
        # 插入pullcommit
        pullcommit_list = list((db['pullcommit'].find({'org': org})))
        for pullcommit in pullcommit_list:
            if int(pullcommit['number']) in pull_set:
                this_db['pullcommit'].insert(pullcommit)
        this_db['pullfile'].create_index([('org', pymongo.ASCENDING),
                                          ('number', pymongo.ASCENDING),
                                          ('sha', pymongo.ASCENDING),
                                          ('commit_at', pymongo.ASCENDING)])

        # 插入pullcomment
        pullcomment_list = list((db['pullcomment'].find({'org': org})))
        for pullcomment in pullcomment_list:
            if int(pullcomment['number']) in pull_set:
                this_db['pullcomment'].insert(pullcomment)
        this_db['pullcomment'].create_index([('org', pymongo.ASCENDING),
                                           ('number', pymongo.ASCENDING),
                                           ('created_at', pymongo.ASCENDING),
                                           ('commentor', pymongo.ASCENDING)])

        # 插入commit
        commit_list = list((db['commit'].find({'org': org, 'commit_at': {'$lt': constants.STMAP_2016_7_31}})))
        for commit in commit_list:
            this_db['commit'].insert(commit)
        this_db['commit'].create_index([('org', pymongo.ASCENDING),
                                        ('author', pymongo.ASCENDING),
                                        ('committer', pymongo.ASCENDING),
                                        ('commit_at', pymongo.ASCENDING)])

