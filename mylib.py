# coding: UTF-8

import math
import cmath
import random

import scipy.linalg as slinalg
import numpy.linalg as linalg
import numpy as np


#------------------------------------
# function definition
#------------------------------------
# for complex256 matrix condition number
def cond(a):
  b = np.array(a, dtype=complex)
  return linalg.cond(b)

# for float128 matrix condition number
def singular(a):
  b = np.array(a, dtype=np.float64)
  u,s,v = np.linalg.svd(b)
  return s


#------------------------------------
# test code
#------------------------------------
def generate_random_matrix(size, val_range):
  r = np.zeros((size,size), dtype=np.complex256)
  for i in range(0,size):
    for j in range(0,size):
      r[i,j] = random.uniform(-val_range,val_range)
  return r


#test = (generate_random_matrix(10, 10))
#print(cond(test))
