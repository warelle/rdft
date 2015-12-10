# coding: UTF-8

import math
import cmath

import scipy.linalg as slinalg
import numpy.linalg as linalg
import numpy as np


#------------------------------------
# function definition
#------------------------------------
def identity(size):
  r = np.zeros((size,size))
  for i in range(0,size):
    r[i,i] = 1.0
  return r
def random(size, val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      r[i,j] = random.uniform(-val_range,val_range)
  return r

def random_well_condition(size, val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  while True:
    for i in range(0,size):
      for j in range(0,size):
        r[i,j] = random.uniform(-val_range,val_range)
    if linalg.cond(r) < 10000:
      break
  return r

def upper_triangle(size, val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      if i <= j:
        r[i,j] = random.uniform(-val_range, val_range)
  return r
def lower_triangle(size, val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      if i >= j:
        r[i,j] = random.uniform(-val_range, val_range)
  return r

def diag_big(size, big_val_range, small_val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      if i == j:
        r[i,j] = random.uniform(-big_val_range, big_val_range)
      else:
        r[i,j] = random.uniform(-small_val_range, small_val_range)
  return r

def diag_zero(size, val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      if i == j:
        r[i,j] = 0.0
      else:
        r[i,j] = random.uniform(-val_range, val_range)
  return r

def cauchy(size, val_range):
  import random
  x,y = [],[]
  for i in range(0,size):
    x.append(random.uniform(-val_range,val_range))
    y.append(random.uniform(-val_range,val_range))
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      r[i,j] = 1.0/(x[i] - y[j])
  return r

def hilbert(size):
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      r[i,j] = 1.0/(i+j+1)
  return r

def z_matrix(size, val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      if i == j:
        r[i,j] = random.uniform(-val_range, val_range)
      else:
        r[i,j] = random.uniform(-val_range, 0)
  return r

def ascending_vector_matrix(size):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      if i == 0:
        r[i,j] = random.uniform(-100,0)
      else:
        r[i,j] = r[i-1,j] + random.uniform(0,5)
  return r

def toeplitz(size,val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      if i == 0 or j == 0:
        r[i,j] = random.uniform(-val_range,val_range)
      else:
        r[i,j] = r[i-1,j-1]
  return r

def circular(size,val_range,opt):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    if 0 <= i and i < opt:
      r[i,0] = random.uniform(-val_range,val_range)
  for i in range(1,size):
    r[0,i] = r[size-i,0]
  for i in range(1,size):
    for j in range(1,size):
      r[i,j] = r[i-1,j-1]
  return r

def circular_partitioned(size,val_range,opt):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  rnums = []
  i = 0
  while i<opt:
    rnum = random.randint(0,size-1)
    continue_flg = False
    for j in range(0,len(rnums)):
      if rnums[j] == rnum:
        continue_flg = True
    if continue_flg:
      continue
    rnums.append(rnum)
    r[rnum,0] = random.uniform(-val_range,val_range)
    i = i + 1
  for i in range(1,size):
    r[0,i] = r[size-i,0]
  for i in range(1,size):
    for j in range(1,size):
      r[i,j] = r[i-1,j-1]
  print(rnums)
  return r

def circular_shift(size,val_range,opt,shift):
  import random
  r  = np.zeros((size,size), dtype=np.complex128)
  rs = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    if 0 <= i and i < opt:
      r[i,0] = random.uniform(-val_range,val_range)
  for i in range(1,size):
    r[0,i] = r[size-i,0]
  for i in range(1,size):
    for j in range(1,size):
      r[i,j] = r[i-1,j-1]
  for i in range(0,size):
    for j in range(0,size):
      if i-shift >= 0:
        rs[i,j] = r[i-shift,j]
      else:
        rs[i,j] = r[size-shift+i,j]
  return rs

def circular_like(size,val_range,opt):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    if 0 <= i and i < opt:
      r[i,0] = random.uniform(-val_range,val_range)
  for i in range(1,size):
    for j in range(1,size):
      r[i,j] = r[i-1,j-1]
  return r

def sparse1(size, val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  half = size/2
  a,b,c,d = random.uniform(-val_range,val_range),random.uniform(-val_range,val_range),random.uniform(-val_range,val_range),random.uniform(-val_range,val_range)
  for i in range(0,half):
    r[i,i] = a
    r[half+i,i] = b
    r[i,half+i] = c
    r[half+i,half+i] = d
  return r

def sparse2(size, val_range, dist):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  half = size/2
  a,b,c,d = random.uniform(-val_range,val_range),random.uniform(-val_range,val_range),random.uniform(-val_range,val_range),random.uniform(-val_range,val_range)
  for i in range(0,half):
    r[i,i] = a
    r[dist+i,i] = b
    r[i,half+i] = c
    r[half+i,half+i] = d
  return r

def sparse3(size, val_range, dist):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  half = size/2
  a,b,c,d = random.uniform(-val_range,val_range),random.uniform(-val_range,val_range),random.uniform(-val_range,val_range),random.uniform(-val_range,val_range)
  for i in range(0,half):
    r[i,i] = a
    r[half+i,i] = b
    r[i,dist+i] = c
    r[half+i,half+i] = d
  return r

def arrowhead(size,val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  r[0,0] = random.uniform(-val_range,val_range)
  for i in range(1,size):
    r[i,i] = random.uniform(-val_range,val_range)
    r[i,0] = random.uniform(-val_range,val_range)
    r[0,i] = random.uniform(-val_range,val_range)
  return r

def arrowbottom(size,val_range):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    r[i,i] = random.uniform(-val_range,val_range)
    r[i,size-1] = random.uniform(-val_range,val_range)
    r[size-1,i] = random.uniform(-val_range,val_range)
  return r

def obi1(size, val_range, opt):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  r[0,0] = random.uniform(-val_range,val_range)
  for i in range(1,opt):
    r[i,0] = random.uniform(-val_range,val_range)
    r[0,i] = random.uniform(-val_range,val_range)
  for i in range(1,size):
    for j in range(1,size):
      r[i,j] = r[i-1,j-1]
  return r
def obi2(size, val_range, opt):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  r[0,0] = random.uniform(-val_range,val_range)
  for i in range(1,opt):
    r[i,0] = random.uniform(-val_range,val_range)
    r[0,i] = random.uniform(-val_range,val_range)
  for i in range(1,size):
    for j in range(1,size):
      if r[i-1,j-1] != 0:
        r[i,j] = random.uniform(-val_range,val_range)
  return r
def obi3(size, val_range, opt):
  import random
  r = np.zeros((size,size), dtype=np.complex128)
  r[0,0] = random.uniform(-val_range,val_range)
  for i in range(1,opt):
    r[i,0] = random.uniform(-val_range,val_range)
    r[0,i] = random.uniform(-val_range,val_range)
  for i in range(1,size):
    if r[size-i,0] != 0:
      r[0,i] = random.uniform(-val_range,val_range)
      r[i,0] = random.uniform(-val_range,val_range)
  for i in range(1,size):
    for j in range(1,size):
      if r[i-1,j-1] != 0:
        r[i,j] = random.uniform(-val_range,val_range)
  return r



#------------------------------------
# test code
#------------------------------------
