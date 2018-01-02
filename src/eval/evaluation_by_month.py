#  计算ACC，并输出每月的ACC，并汇总结果到csv
import src.database.dbutil as dbutil
import src.constants as constants
import csv

WHOLE_ACC = {}
if __name__ == '__main__':
    csvfile = open(constants.GOUSIOS_CODE_PATH+'/result/gousios.csv', 'w', encoding='utf8', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['gousios', 'ACC'])

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

        infile_result = open(constants.GOUSIOS_DATA_PATH+'/'+org+'/result.txt', "r")
        outfile_month = open(constants.GOUSIOS_DATA_PATH+'/'+org+'/eval-month.txt', "w")

        resultDict = {}
        for line in infile_result.readlines():
            newline = line.strip('\r\n').split(',')
            if float(newline[1]) >= 0.5:
                resultDict[int(newline[0])] = True
            else:
                resultDict[int(newline[0])] = False

        resultMonthDict = {}
        for pull in resultDict:
            month = actual_merged_dict[pull]['month']
            number = actual_merged_dict[pull]['number']
            merged = actual_merged_dict[pull]['merged']
            if month not in resultMonthDict:
                resultMonthDict[month] = {}
                resultMonthDict[month][number] = 0 if resultDict[number] ^ merged else 1
            else:
                resultMonthDict[month][number] = 0 if resultDict[number] ^ merged else 1

        # 整合每个月份,顺便输出文件
        sorted_result = []
        for month in resultMonthDict:
            ACC = 0.0
            for number in resultMonthDict[month]:
                ACC += resultMonthDict[month][number]
            ACC = ACC / len(resultMonthDict[month])
            sorted_result.append(str(month) + ':' + str(ACC) + '\n')

        sorted_result.sort(key=lambda x: int(x.split(':')[0]))
        for result in sorted_result:
            outfile_month.write(result)

        AVE_ACC = 0.0
        for result in sorted_result:
            AVE_ACC += float(result.strip('\r\n').split(':')[1])
        AVE_ACC = AVE_ACC / len(sorted_result)
        WHOLE_ACC[org] = AVE_ACC
    for org, repo in constants.org_list:
        writer.writerow([org, WHOLE_ACC[org]])
    csvfile.flush()
    csvfile.close()









