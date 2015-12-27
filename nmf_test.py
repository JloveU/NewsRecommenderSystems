#!python3
"""
Created on 2015-12-12
@author: yuqiang
Test of module "nmf"
"""

import numpy
import time
import nmf

V = [
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ]
V = [
        [1, 1, 0, 1, 1],
        [1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
    ]

V = numpy.array(V)
print("V = ")
print(V)

time_start = time.time()
K = 2
W, H = nmf.nmf(V, K)
time_end = time.time()
estimatedV = numpy.dot(W, H)
print("W = ")
print(W)
print("H = ")
print(H)
print("estimatedV = ")
print(estimatedV)
print(time_end - time_start)
