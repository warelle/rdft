# coding: UTF-8

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
def direct_solver(a,b):
  return slinalg.solve(a,b)
def direct_lu_solver(a,b):
  lup = slinalg.lu_factor(a)
  return slinalg.lu_solve(lup,b)

def plu(a,b):
  (size,_) = a.shape
  (lu,p) = slinalg.lu_factor(a)
  l = np.zeros((size,size), dtype=np.complex128)
  u = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    l[i,i] = 1
    for j in range(0,size):
      if i>j:
        l[i,j] = lu[i,j]
      else:
        u[i,j] = lu[i,j]
  c = np.zeros(size, dtype=np.complex128)
  for i in range(0,size):
    c[i] = b[i]
  for i in range(0,size):
    tmp = c[i]
    c[i] = c[p[i]]
    c[p[i]] = tmp
  return (l, u, c)

def lu_solver(a,b):
	(l,u,c) = plu(a,b)
	y     = lu.l_step(l,c)
	x     = lu.u_step(u,y)
	return x


#------------------------------------
# test code
#------------------------------------

