'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
'''
method = "gousios_new"

import csv
from src.database import dbutil
from src.constants import org_list
if __name__ == '__main__':
    client = dbutil.get_connection()
    L = []
    for org, repo in org_list:
        result = client[org]['result'].find_one()['gousios_new']
        L.append(org+','+str(result))
    L.sort(key=lambda x: float(x.split(',')[1]), reverse=True)
    outfile = open("data/"+method+".csv", "w", newline="")
    writer = csv.writer(outfile)
    writer.writerow(['org', 'acc'])
    for org in L:
        writer.writerow(org.split(','))
    dbutil.close_connection(client)