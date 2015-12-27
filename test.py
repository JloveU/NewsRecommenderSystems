#!python3
"""
test
"""


'''
a = [2, 3, 3, 1, 1, 4, 5]
b = [x for x in a if x < 3]
inds = [i for (i, val) in enumerate(a) if val < 3]
c = [(i, val) for (i, val) in enumerate(a)]
c.sort(key=lambda x: x[1])
'''

'''
import time
def linecount_1( ):
    return len(open("user_click_data.txt", "r", 1, "utf-8").readlines( ))
def linecount_2( ):
    count = -1
    for count, line in enumerate(open("user_click_data.txt", "r", 1, "utf-8")): pass
    return count+1
def linecount_3( ):
    count = 0
    thefile = open("user_click_data.txt", "r", 1, "utf-8")
    while True:
        buffer = thefile.read(65536)
        if not buffer: break
        count += buffer.count('\n')
    return count
time_start = time.time()
for i in list(range(10)):
    linecount_1()
time_end = time.time()
print("linecount_1: %f  count=%d" % (time_end - time_start, linecount_1()))
time_start = time.time()
for i in list(range(10)):
    linecount_2()
time_end = time.time()
print("linecount_2: %f  count=%d" % (time_end - time_start, linecount_2()))
time_start = time.time()
for i in list(range(10)):
    linecount_3()
time_end = time.time()
print("linecount_3: %f  count=%d" % (time_end - time_start, linecount_3()))
'''

"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import codecs
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

user_ids = []
news_ids = []
time = []
news_title = []
news_body = []

def get_filedata(filename):
    try:
        with open(filename, "r", 1, "utf-8") as f:   #with sentence open and close file automatically
            data = f.readline()
            print(data)
            #print data.split('\t')
            sp = data.split('\t')
            user_ids.append(sp[0])
            news_ids.append(sp[1])
            time.append(sp[2])
            news_title.append(sp[3])
            print(news_title[-1])
            news_body.append(sp[4])
            print(news_body[-1])
    except IOError as ioerr:
        print('File Error' + str(ioerr))    #print the error
        return None

get_filedata("user_click_data.txt")
"""

"""
import time
import sys

time_start = time.time()
for i in list(range(100)):
    print(".", end="")
time_end = time.time()
print(time_end - time_start)

time_start = time.time()
for i in list(range(100)):
    sys.stdout.write('.'); sys.stdout.flush()
time_end = time.time()
print(time_end - time_start)
"""

"""
import numpy
a = numpy.zeros((2, 2))
a.dump("a.numpydumpedarray")
"""

"""
import numpy
a = numpy.array([[1,2],[3,4],[2,4],[3,2],[2,4]])
b = a.tolist()
b_key = [x for [x, y] in b]
b_key_dict = {value: index for (index, value) in enumerate(b_key)}
unique_ids = b_key_dict.values()
c = [b[index] for index in unique_ids]
"""

"""
# -*- coding:utf-8 -*-
#!python3

import math
import numpy as np
import os.path
import time

time_start = time.time()
dim = 9
def file_news_id(filename):
    '''
    filename:
    '''
    fr = open(filename)
    train_news = []

    while True:
        line = fr.readline()
        if line:
            p = line.split('\n')
            train_news.append(int(p[0]))
        else:
            break


    return train_news

def file_news_wordlist(filename):
    '''
    filename:
    '''
    fr = open(filename,'r', 1, "utf-8")
    wordlist = []

    while True:
        line = fr.readline()
        if line:
            p = line.split('\n')
            wordlist.append(p[0])
        else:
            break


    return wordlist

s = os.getcwd()
train_news_id = file_news_id('train_id.txt')
train_wordlist = file_news_wordlist('frequence_word_use.txt')
time_end = time.time()
print(time_end - time_start)                   #prepare labels return
"""

"""
import numpy
eps = numpy.finfo(float).eps
"""

"""
import numpy
a = numpy.array([[0,1,1,0], [1,1,0,1]])
b = numpy.zeros((2, 4))
b[a==1] = 1
"""

"""
import numpy
a = numpy.array([[0,1,1,0], [1,1,0,1]])
b = numpy.zeros((2, 4))
c = a[0] * 0.5 + b[1]
"""

"""
import numpy
a = numpy.array([0,2,3,2,1])
index = numpy.argsort(a)
"""

"""
import numpy
a = numpy.array([1,2,3,4,5])
inverse_indexes = range(4, -1, -1)
b = a[inverse_indexes]
"""

"""
import numpy
a = numpy.array([1,2,3])
b = numpy.array([4,5,6])
c = numpy.row_stack((a, b))
d = numpy.column_stack((a, b))
"""

"""
# calculate similarity between users
print("Calculate similarity between users started.")
time_start = time.time()
user_user_similarities = numpy.zeros((user_num, user_num), numpy.float16)
# progress_count = 0
# for i in range(0, user_num):
#     for j in range(i + 1, user_num):  # similarity to oneself is set to "0"
#         similarity = H[:, i].dot(H[:, j])
#         user_user_similarities[i, j] = similarity
#         user_user_similarities[j, i] = similarity
#         # progress_count += 1
#         # if progress_count % 100000 == 0:
#         #     print("%f %%" % (progress_count*2 / (user_num*(user_num-1)) * 100))
#         #     # sys.stdout.write("\r%f %%\r" % (progress_count*2 / (user_num*(user_num-1)) * 100)); sys.stdout.flush()
#     if i % 10 == 0:
#         print("%f%%. %f s elapsed." % (i / user_num * 100, time.time() - time_start))
H_transpose = H.transpose()
computed_count = 0
compute_step = 1000  # to avoid MemoryError, only compute a part each time
while computed_count < user_num:
    compute_upper_limit = min((computed_count + compute_step, user_num))
    user_user_similarities[computed_count:compute_upper_limit, :] = numpy.dot(H_transpose[computed_count:compute_upper_limit, :], H)
    computed_count += compute_step
time_end = time.time()
print("Calculate similarity between users ended. %f s cost." % (time_end - time_start))
# numpy.save("user_user_similarities.npy", user_user_similarities)
# user_user_similarities = numpy.load("user_user_similarities.npy")
"""

"""
import numpy
a = numpy.eye(3, dtype=numpy.bool_)
b = numpy.int16(a)
c = a[0] * 3
b[a == 1] = 2
"""

"""
import numpy
import scipy.io
user_news_recommend_indexes = numpy.load("user_news_recommend_indexes.npy")
scipy.io.savemat("user_news_recommend_indexes.mat", {"user_news_recommend_indexes": user_news_recommend_indexes})
# user_ids = numpy.load("user_ids.npy")
# scipy.io.savemat("user_ids.mat", {"user_ids": user_ids})
# news_ids = numpy.load("news_ids.npy")
# scipy.io.savemat("news_ids.mat", {"news_ids": news_ids})
"""

"""
import numpy
a = numpy.array([[1,2,3], [1], [1,2,3,4]])
b = numpy.array([[1,2,3,5], [1,6,7,8], [1,2,3,4]])
"""

"""
import data
# news_dict = data.get_news_dict()
# user_clicked_news_dict = data.get_user_clicked_news_dict()
user_news_array_for_train, user_ids_for_train, news_ids_for_train, user_news_array_for_test, user_ids_for_test, news_ids_for_test = data.get_user_news_arrays_of_train_and_test()
"""

"""
a = list(range(5)) + list(range(5))
for b in a:
    if b % 2 == 0:
        a.remove(b)

title_cuts_without_stop_word = [word for word in title_cuts if word not in stopkeys]
"""

"""
import numpy
a = numpy.array([[1,2], [3,4]])
# a.tofile("a.bin")
# a.tofile("a.bin", sep=" ")
numpy.savetxt("a.txt", a, fmt='%d', delimiter='\t')
b = numpy.loadtxt("a.txt", dtype=numpy.int16, delimiter='\t')
a = numpy.array([[0.1,2.2], [3.3,4]])
numpy.savetxt("a.txt", a, fmt='%.6f', delimiter='\t')
b = numpy.loadtxt("a.txt", dtype=numpy.float32, delimiter='\t')
"""

"""
import data
user_news_array_for_train, user_ids_for_train, news_ids_for_train, user_news_array_for_test, user_ids_for_test, news_ids_for_test = data.get_user_news_arrays_of_train_and_test(True, True)
"""

"""
import numpy
a = numpy.array([[1, 2], [3, 4]])
b = a.transpose()
"""

"""
import sys
import time
bar_length = 20
for percent in range(0, 101):
    hashes = '#' * int(percent/100.0 * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [%s] %d%%"%(hashes + spaces, percent))
    sys.stdout.flush()
    time.sleep(0.1)
"""

print("")
