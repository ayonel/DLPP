# 一些工具函数
import datetime
import time
import csv
import psutil


# 传入两个格式化的时间字符串，输出两个时间之间的天数
def datediff(beginDate,endDate):
    beginDate = beginDate.split("T")[0]
    bd_year = int(beginDate.split("-")[0])
    bd_month = int(beginDate.split("-")[1])
    bd_day = int(beginDate.split("-")[2])

    endDate = endDate.split("T")[0]
    ed_year = int(endDate.split("-")[0])
    ed_month = int(endDate.split("-")[1])
    ed_day = int(endDate.split("-")[2])
    bd = datetime.date(bd_year, bd_month, bd_day)
    ed = datetime.date(ed_year, ed_month, ed_day)
    return (ed-bd).days


# 传入一个格式化的时间字符串，和一个时间戳，输出之间的天数
def datediff_stamp(beginDate,endStamp):
    begin_stamp = time.mktime(time.strptime(beginDate, '%Y-%m-%dT%H:%M:%SZ'))
    return int((endStamp-begin_stamp)/(24*3600))


# 获取毫秒数
def getTimestamp(str):
    return datetime.datetime.strptime(str, '%Y-%m-%dT%H:%M:%SZ').timestamp()


# 指定一个最大时间值,暂定为当前时间
def getMaxTimestamp():
    return 14761738870.0

def getCSVWriter(filename, op):
    outfile = open(filename, op, newline="")
    writer = csv.writer(outfile)
    return writer


def memory_available():
    memory = psutil.virtual_memory()
    return memory.available / memory.total


def getTodayString():
    return datetime.datetime.now().strftime("%Y-%m-%d")


def iandfrange(start, *args):
    """ 
    输入：函数可接收最多三个参数，依次分别是起始值，结束值和步长，可以做任意整数和小数的range功能 
    输出：返回值为包含起始值的，以起始值迭代加步长，直到最后一个<=结束值的值为止的一个列表 
    约定： 
        1.如果只传入2个参数，自动匹配给起始值和结束值，步长默认为1 
        2.如果只传入1个参数，自动匹配给结束值，起始值默认为0，步长默认为1 
    测试： 
        1.参数超过3个 
        2.参数传入2个 
        3.参数传入1个 
        4.步长传入的值非整数和小数的情况 
        5.start>=end的情况 
        6.用户输入的起始值加步长经过1次计算后即超过结束值的情况 
    声明： 
        1.程序仅供学习使用，实现并未参考numpy.arange的实现，测试如有不详尽之处，请多多指教 
        2.程序中对计算结果如果不四舍五入，可能会得到类似"2.5366600000000004"这样精度有错误的结果， 
          此结果由计算机本身的精度误差导致 
    """
    # 保证传入参数不超过3个，超过则报错提示用户
    try:
        args[2]
    except Exception as e:
        pass
    else:
        raise Exception(ValueError, "The function receive three args!")
        # 保证传入的3个参数能正确匹配到start,end和step三个变量上
    try:
        end, step = args[0], args[1]
    except IndexError:
        try:
            end = args[0]
        except IndexError:
            end = start
            start = 0
        finally:
            step = 1
            # 参数正确性校验，包括对step是否是int或float的校验，提示用户输出数据可能只有start的校验以及start>=end的情况
    try:
        try:
            a, b = str(step).split(".")
            roundstep = len(b)
        except Exception as e:
            if isinstance(step, int):
                roundstep = 0
            else:
                raise Exception(TypeError, "Sorry,the function not support the step type except integer or float!")
        if start + step >= end:
            print("The result list may include the 'start' value only!")
        if start >= end:
            raise Exception(ValueError,
                            "Please check you 'start' and 'end' value,may the 'start' greater or equle the 'end'!")
    except TypeError as e:
        print(e)
    else:
        pass
        # 输出range序列
    lista = []
    while start < end:
        lista.append(start)

        start = round(start + step, roundstep)
    return lista


def getWeekFromStamp(timestamp):
    time_local = datetime.datetime.utcfromtimestamp(timestamp)
    return time_local.weekday()

