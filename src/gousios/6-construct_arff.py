# 将gousios论文中的各项属性，构造成arff文件，用于weka计算。
# 构造出来的格式为，number,month,is_merged,各项属性


import os
import pymongo
import src.constants as constants
import src.database.dbutil as dbutil
gousios_attr_num = 15

if __name__ == '__main__':
    client = dbutil.get_connection()
    for org, repo in constants.org_list:
        if not os.path.exists(constants.GOUSIOS_DATA_PATH + '/'+org):
            os.mkdir(constants.GOUSIOS_DATA_PATH + '/'+org)

        outfile_data_arff = open(constants.GOUSIOS_DATA_PATH + '/' + org + '/data.arff', 'w')
        outfile_empty_arff = open(constants.GOUSIOS_DATA_PATH + '/' + org + '/empty.arff', 'w')

        outfile_data_arff.write('@relation \'Merge -C 2\'\n\n')
        outfile_data_arff.write('@attribute number numeric\n')
        outfile_data_arff.write('@attribute month numeric\n')
        outfile_data_arff.write('@attribute merged {0,1}\n')

        outfile_empty_arff.write('@relation \'Merge -C 2\'\n\n')
        outfile_empty_arff.write('@attribute number numeric\n')
        outfile_empty_arff.write('@attribute month numeric\n')
        outfile_empty_arff.write('@attribute merged {0,1}\n')


        # -----------------------------开始描述各项属性--------------------------------
        #############################PULL REQUEST##################################
        outfile_data_arff.write('@attribute num_commits numeric\n')
        outfile_data_arff.write('@attribute src_churn numeric\n')
        outfile_data_arff.write('@attribute test_churn numeric\n')
        outfile_data_arff.write('@attribute files_changes numeric\n')
        # outfile_data_arff.write('@attribute num_comments numeric\n')
        # outfile_data_arff.write('@attribute num_particpants numeric\n')
        # outfile_data_arff.write('@attribute conflict {0,1}\n')
        # outfile_data_arff.write('@attribute forward_link {0,1}\n')
        #############################PROJECT##################################
        outfile_data_arff.write('@attribute sloc numeric\n')
        outfile_data_arff.write('@attribute team_size numeric\n')
        outfile_data_arff.write('@attribute perc_ext_contribs numeric\n')
        outfile_data_arff.write('@attribute commits_files_touched numeric\n')
        outfile_data_arff.write('@attribute test_lines_per_kloc numeric\n')
        #############################DEVELOPER##################################
        outfile_data_arff.write('@attribute prev_pullreqs numeric\n')
        outfile_data_arff.write('@attribute requester_succ_rate numeric\n')


        #############################PULL REQUEST##################################
        outfile_empty_arff.write('@attribute num_commits numeric\n')
        outfile_empty_arff.write('@attribute src_churn numeric\n')
        outfile_empty_arff.write('@attribute test_churn numeric\n')
        outfile_empty_arff.write('@attribute files_changes numeric\n')
        # outfile_empty_arff.write('@attribute num_comments numeric\n')
        # outfile_empty_arff.write('@attribute num_particpants numeric\n')
        # outfile_empty_arff.write('@attribute conflict {0,1}\n')
        # outfile_empty_arff.write('@attribute forward_link {0,1}\n')
        #############################PROJECT##################################
        outfile_empty_arff.write('@attribute sloc numeric\n')
        outfile_empty_arff.write('@attribute team_size numeric\n')
        outfile_empty_arff.write('@attribute perc_ext_contribs numeric\n')
        outfile_empty_arff.write('@attribute commits_files_touched numeric\n')
        outfile_empty_arff.write('@attribute test_lines_per_kloc numeric\n')
        #############################DEVELOPER##################################
        outfile_empty_arff.write('@attribute prev_pullreqs numeric\n')
        outfile_empty_arff.write('@attribute requester_succ_rate numeric\n')

        outfile_data_arff.write('\n@data\n\n')
        outfile_empty_arff.write('\n@data\n\n')

        # 读取gousios信息
        gousios_list = list(client[org]['gousios'].find())
        gousios_list.sort(key=lambda x: int(x['number']))

        # 读取原始pull信息，获取是否merged
        pullinfo_list = list(client[org]['pullinfo'].find())
        merged_dict = {}
        for pullinfo in pullinfo_list:
            if pullinfo['merged']:
                merged_dict[str(pullinfo['number'])] = '1'
            else:
                merged_dict[str(pullinfo['number'])] = '0'

        for pull in gousios_list:
            if pull['conflict']:
                conflict = '1'
            else:
                conflict = '0'

            if pull['forward_link']:
                forward_link = '1'
            else:
                forward_link = '0'

            string = str(pull['number'])+','+str(pull['month'])+','+merged_dict[pull['number']]+','\
                + str(pull['num_commits'])+',' \
                + str(pull['src_churn'])+',' \
                + str(pull['test_churn'])+',' \
                + str(pull['files_changes']) + ',' \
                + str(pull['sloc']) + ',' \
                + str(pull['team_size']) + ',' \
                + str(pull['perc_ext_contribs']) + ',' \
                + str(pull['commits_files_touched']) + ',' \
                + str(pull['test_lines_per_kloc']) + ',' \
                + str(pull['prev_pullreqs']) + ',' \
                + str(pull['requester_succ_rate']) + '\n'
            outfile_data_arff.write(string)

