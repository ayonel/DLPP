org_list = [
    ('adobe', 'brackets'),
    ('angular', 'angular.js'),
    ('apache', 'spark'),
    ('Baystation12', 'Baystation12'),
    ('bitcoin', 'bitcoin'),
    ('cakephp', 'cakephp'),
    ('caskroom', 'homebrew-cask'),
    ('cdnjs', 'cdnjs'),
    ('cocos2d', 'cocos2d-x'),
    ('dimagi', 'commcare-hq'),
    ('dlang', 'dmd'),
    ('hrydgard', 'ppsspp'),
    ('ipython', 'ipython'),
    ('joomla', 'joomla-cms'),
    ('Katello', 'katello'),
    ('nodejs', 'node-v0.x-archive'),
    ('openmicroscopy', 'openmicroscopy'),
    ('owncloud', 'core'),
    ('puppetlabs', 'puppet'),
    ('rails', 'rails'),
    ('rapid7', 'metasploit-framework'),
    ('scala', 'scala'),
    ('symfony', 'symfony'),
    ('sympy', 'sympy'),
    ('tgstation', 'tgstation'),
    ('TrinityCore', 'TrinityCore'),
    ('twbs', 'bootstrap'),
    ('wet-boew', 'wet-boew'),
    ('Wikia', 'app'),
    ('zendframework', 'zendframework')
]
from urllib import request
import json
for org,repo in org_list:
    url = 'https://api.github.com/repos/'+org+'/'+repo+'/pulls?state=all&direction=asc&access_token=d147a86290a1828205ec287d0c722ffbd66cd4f9'
    data = request.urlopen(url).read().decode("utf8")
    # print('\''+org+'\':\''+json.loads(data)[0]['created_at']+'\',')
    print(json.loads(data)[0]['created_at'].split('T')[0])