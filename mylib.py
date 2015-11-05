# coding: UTF-8

import math
import cmath
import random

import scipy.linalg as slinalg
import numpy.linalg as linalg
import numpy as np

import ldsin

#------------------------------------
# function definition
#------------------------------------
# for complex256 matrix condition number
def cond(a):
  b = np.array(a, dtype=complex)
  return linalg.cond(b)

# for float128 matrix singular value
def singular(a):
  b = np.array(a, dtype=np.float64)
  u,s,v = np.linalg.svd(b)
  return s

# for float128 matrix singular value
def complex_singular(a):
  b = np.array(a, dtype=np.complex128)
  u,s,v = np.linalg.svd(b)
  return s

# casting float128=>float64=>float128
def float128_64_128(x):
  y = np.array(x, dtype=np.float64)
  return np.array(y, dtype=np.float128)

# casting complex256=>complex128=>complex256
def complex256_128_256(x):
  y = np.array(x, dtype=np.complex128)
  return np.array(y, dtype=np.complex256)

#----------------------------
# below is sin,cos of float128
# maybe fail by ordinary idea
#----------------------------

# pi
def pi():
  r = np.array([3.14159265358979323846264338327950288419716939937],dtype=np.float128)
  return r

# return array e^ix with array scalar
def exp_i(x):
  r = np.array((1.),dtype=np.complex256)
  r = ldsin.ldcos(x) + 1j*ldsin.ldsin(x)
  return r

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
#r = np.array([cmath.sin(1.0)])

#tmp = 100.0
#ee = exp_i(tmp)
#print(ee, cmath.rect(1,tmp))
#print(linalg.norm(exp_i(tmp) - cmath.rect(1,tmp)))

def generate_f(size):
  r = np.zeros((size,size), dtype=np.complex256)
  for i in range(0,size):
    for j in range(0,size):
      r[i,j] = (cmath.rect(1,(-2.0*math.pi/size)*i*j))/cmath.sqrt(size)
  return np.array(r,dtype=np.complex256)
def generate_f_m(size):
  r = np.zeros((size,size), dtype=np.complex256)
  for i in range(0,size):
    for j in range(0,size):
      t = exp_i((-2.0*math.pi/size)*i*j)
      tmp = t/cmath.sqrt(size)
      r[i,j] = tmp
  return r

#print(generate_f(100) - generate_f_m(100))
#a = np.array([200],dtype=np.float64)
#b = np.array([200],dtype=np.float128)
#n = ldsin.ldsin(b)
#print( "%.100f" % math.sin(a[0]) )
#print( "%.100f" % n[0])
