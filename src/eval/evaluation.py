#  计算ACC，并输出每月的ACC，并汇总结果到csv
import src.database.dbutil as dbutil
import src.constants as constants
import csv

ALGO = 'gousios_new'

WHOLE_ACC = {}
if __name__ == '__main__':
    client = dbutil.get_connection()
    csvfile = open(constants.GOUSIOS_CODE_PATH+'/result/gousios_simple.csv', 'w', encoding='utf8', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['gousios_simple', 'ACC'])

    client = dbutil.get_connection()

    for org, repo in constants.org_list:
        # 实际是否关闭的字典,键为number
        actual_merged_dict = {}
        pullinfo_list = list(client[org]['pullinfo'].find())
        pullinfo_list.sort(key=lambda x: int(x['number']))

        for pullinfo in pullinfo_list:
            actual_merged_dict[pullinfo['number']] = {}
            actual_merged_dict[pullinfo['number']]['number'] = pullinfo['number']
            actual_merged_dict[pullinfo['number']]['month'] = pullinfo['month']
            actual_merged_dict[pullinfo['number']]['merged'] = pullinfo['merged']

        infile_result = open(constants.GOUSIOS_DATA_PATH+'/'+org+'/result_simple.txt', "r")

        resultDict = {}
        for line in infile_result.readlines():
            newline = line.strip('\r\n').split(',')
            if float(newline[1]) >= 0.5:
                resultDict[int(newline[0])] = True
            else:
                resultDict[int(newline[0])] = False

        accurary_pull_num = 0
        for test_pull in resultDict:
            if not resultDict[test_pull] ^ actual_merged_dict[test_pull]['merged']:
                accurary_pull_num += 1
        accuracy = accurary_pull_num / len(resultDict)
        writer.writerow([org, accuracy])
        client[org]['result'].delete_many({'gousios': {'$exists': True}})
        client[org]['result'].delete_many({'gousios_new': {'$exists': True}})
        client[org]['result'].insert({ALGO: accuracy})
    csvfile.flush()
    csvfile.close()
    dbutil.close_connection(client)

    # gousios_new  删去了几个评论属性，只保留pr创建时的属性









