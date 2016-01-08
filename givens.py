#coding: UTF-8

import math
import cmath
import random

import scipy.linalg as slinalg
import numpy.linalg as linalg
import numpy as np

import lu

#------------------------------------
# function definition
#------------------------------------
def one_givens_rotation(size,i,j,theta):
  r = np.identity(size, dtype=np.float64)
  r[i,i] = math.cos(theta)
  r[j,i] = math.sin(theta)
  r[i,j] = -math.sin(theta)
  r[j,j] = math.cos(theta)
  return r

def givens_list_generation(size):
  givens_list = []
  cr_list = []
  for i in xrange(size):
    cr_list.append(i)
  while len(cr_list) > 1:
    rand_num = random.randint(0,len(cr_list)-1)
    i = cr_list.pop(rand_num)
    rand_num = random.randint(0,len(cr_list)-1)
    j = cr_list.pop(rand_num)
    t = random.uniform(0,2*math.pi)
    givens_list.append(one_givens_rotation(size,i,j,t))
  return givens_list

def givens_generation(size):
  r = np.identity(size, dtype=np.float64)
  givens_list = givens_list_generation(size)
  for g in givens_list:
    r = r.dot(g)
  return r

