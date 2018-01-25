ROOT_PATH = "F:/DLPP"
GOUSIOS_DATA_PATH = ROOT_PATH+'/data/gousios_new'
GOUSIOS_CODE_PATH = ROOT_PATH+'/src/gousios'

persistence_db = 'test'
persistence_col = 'pullpredict'

MAX_DAY = 10000 # 用于默认的天数，比如评审时间默认为MAX_DAY

RANDOM_SEED = 1

# 下文注释的6个项目因为pr通过率过高，或者过低，被排除
org_list = [
    ('Baystation12', 'Baystation12'),
    ('bitcoin', 'bitcoin'),
    ('twbs', 'bootstrap'),
    ('adobe', 'brackets'),
    ('cakephp', 'cakephp'),
    ('cdnjs', 'cdnjs'),
    ('cocos2d', 'cocos2d-x'),
    ('owncloud', 'core'),
    ('dlang', 'dmd'),
    ('caskroom', 'homebrew-cask'),
    ('ipython', 'ipython'),
    ('joomla', 'joomla-cms'),
    ('Katello', 'katello'),
    ('rapid7', 'metasploit-framework'),
    ('openmicroscopy', 'openmicroscopy'),
    ('hrydgard', 'ppsspp'),
    ('puppetlabs', 'puppet'),
    ('rails', 'rails'),
    ('scala', 'scala'),
    ('symfony', 'symfony'),
    ('sympy', 'sympy'),
    ('tgstation', 'tgstation'),
    ('zendframework', 'zendframework'),

    #-----------以下为极为不平衡的6个项目-----------
    # ('apache', 'spark'),
    ('nodejs', 'node-v0.x-archive'),
    ('angular', 'angular.js'),
    ('Wikia', 'app'),
    ('wet-boew', 'wet-boew'),
    ('dimagi', 'commcare-hq'),
]

rest_org_list = [
    ('TrinityCore', 'TrinityCore'),
    ('wet-boew', 'wet-boew'),
    ('angular', 'angular.js'),
    ('dimagi', 'commcare-hq'),
    ('nodejs', 'node-v0.x-archive'),
    ('Wikia', 'app'),
]

# 每个项目的第一条issue创建时间
first_issue_time = {
    'Baystation12':'2011-11-07T16:24:52Z',
    'Katello':'2012-04-12T17:53:30Z',
    'owncloud':'2012-08-25T21:46:11Z',
    'joomla':'2011-09-28T01:34:55Z',
    'adobe':'2011-12-08T18:46:22Z',
    'scala':'2011-12-01T10:41:01Z',
    'dlang':'2011-01-26T22:30:51Z',
    'hrydgard':'2012-11-02T16:37:39Z',
    'rapid7':'2011-08-30T19:14:09Z',
    'tgstation':'2012-01-21T18:51:01Z',
    'cdnjs':'2011-02-26T15:51:36Z',
    'wet-boew':'2012-05-11T20:05:07Z',
    'bitcoin':'2010-12-19T16:17:53Z',
    'cakephp':'2010-09-05T19:16:40Z',
    'ipython':'2010-05-10T09:26:02Z',
    'sympy':'2010-09-01T00:15:59Z',
    'puppetlabs':'2010-09-28T12:31:06Z',
    'openmicroscopy':'2011-09-21T11:27:30Z',
    'apache':'2014-02-25T23:57:18Z',
    'nodejs':'2009-06-03T15:17:06Z',
    'angular': '2010-09-02T15:58:28Z',
    'TrinityCore': '2010-12-31T04:54:43Z',
    'caskroom': '2012-03-10T09:12:33Z',
    'cocos2d': '2010-11-20T14:15:55Z',
    'dimagi': '2010-08-10T15:00:38Z',
    'rails': '2009-04-01T04:24:23Z',
    'symfony': '2010-09-01T18:19:53Z',
    'twbs': '2011-08-19T18:53:33Z',
    'Wikia': '2012-08-23T22:32:23Z',
    'zendframework': '2010-08-25T23:41:00Z'
}

first_issue_time_stamp = {
    'scala': 1322707261.0,
    'twbs': 1313751213.0,
    'Baystation12': 1320654292.0,
    'dlang': 1296052251.0,
    'wet-boew': 1336737907.0,
    'cdnjs': 1298706696.0,
    'cakephp': 1283685400.0,
    'rapid7': 1314702849.0,
    'puppetlabs': 1285648266.0,
    'zendframework': 1282750860.0,
    'sympy': 1283271359.0,
    'tgstation': 1327143061.0,
    'symfony': 1283336393.0,
    'hrydgard': 1351845459.0,
    'ipython': 1273454762.0,
    'bitcoin': 1292746673.0,
    'openmicroscopy': 1316575650.0,
    'angular': 1283414308.0,
    'caskroom': 1331341953.0,
    'joomla': 1317144895.0,
    'dimagi': 1281423638.0,
    'Wikia': 1345732343.0,
    'apache': 1393343838.0,
    'adobe': 1323341182.0,
    'rails': 1238531063.0,
    'owncloud': 1345902371.0,
    'TrinityCore': 1293742483.0,
    'cocos2d': 1290233755.0,
    'Katello': 1334224410.0,
    'nodejs': 1244013426.0
}

first_pr_time = {
    'adobe':'2011-12-08T18:46:22Z',
    'angular':'2010-09-08T15:59:32Z',
    'apache':'2014-02-25T23:57:18Z',
    'Baystation12':'2011-11-07T16:24:52Z',
    'bitcoin':'2010-12-20T15:47:21Z',
    'cakephp':'2010-09-05T19:16:40Z',
    'caskroom':'2012-03-10T09:12:33Z',
    'cdnjs':'2011-02-26T15:51:36Z',
    'cocos2d':'2010-11-20T14:15:55Z',
    'dimagi':'2012-03-14T14:00:28Z',
    'dlang':'2011-01-26T22:30:51Z',
    'hrydgard':'2012-11-02T16:37:39Z',
    'ipython':'2010-09-17T22:40:51Z',
    'joomla':'2011-09-28T01:34:55Z',
    'Katello':'2012-04-12T17:53:30Z',
    'nodejs':'2010-08-31T18:28:10Z',
    'openmicroscopy':'2011-09-21T11:27:30Z',
    'owncloud':'2012-08-25T21:46:11Z',
    'puppetlabs':'2010-09-28T12:31:06Z',
    'rails':'2010-09-02T05:34:17Z',
    'rapid7':'2011-11-10T19:26:05Z',
    'scala':'2011-12-01T10:41:01Z',
    'symfony':'2010-09-01T18:19:53Z',
    'sympy':'2010-09-01T00:15:59Z',
    'tgstation':'2012-01-21T18:51:01Z',
    'TrinityCore':'2010-12-31T04:54:43Z',
    'twbs':'2011-08-19T18:53:33Z',
    'wet-boew':'2012-05-25T21:07:06Z',
    'Wikia':'2012-08-23T22:32:23Z',
    'zendframework':'2010-09-04T22:41:01Z'
}



# 2016年7月31号23:59:59的时间戳
STMAP_2016_7_31 = 1469980799.0

# 三个月的秒数
SECOND_3_MONTH = 3*30*24*3600
SECOND_1_MONTH = 1*30*24*3600
SECOND_1_DAY = 24*3600
SECOND_1_HOUR = 3600

# bot reviewer
bot_reviewer = {
    'Baystation12': ['bs12-bot'],
    'dlang': ['klickverbot'],
    'joomla': ['joomla-cms-bot', 'jissues-bot'],
    'rails': ['rails-bot']
}

bug_word_list = [
    "fix", "error", "bug", " fix", "issue", "mistake", "incorrect", "fault", "defect", "flaw", "patch"
]

week_feature_dict = {
    0: [1, 0, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0, 0],
    2: [0, 0, 1, 0, 0, 0, 0],
    3: [0, 0, 0, 1, 0, 0, 0],
    4: [0, 0, 0, 0, 1, 0, 0],
    5: [0, 0, 0, 0, 0, 1, 0],
    6: [0, 0, 0, 0, 0, 0, 1],
}
