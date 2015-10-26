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

def lu(mat):
  (size, _) = mat.shape
  l, u = np.zeros((size,size), dtype=np.complex128), np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size-1):
    for j in range(i+1,size):
      mat[j,i] = mat[j,i]/mat[i,i]
      for k in range(i+1,size):
        mat[j,k] = mat[j,k] - mat[j,i]*mat[i,k]
  for i in range(0,size):
    l[i,i] = 1
    for j in range(0,size):
      if i>j:
        l[i,j] = mat[i,j]
      else:
        u[i,j] = mat[i,j]
  return (l,u)

def l_step(l, b):
  (size, _) = l.shape
  y = b
  for i in range(0,size):
    for j in range(0,i):
      y[i] -= l[i][j]*y[j]
    y[i] /= l[i][i]
  return y

def u_step(u, y):
  (size,_) = u.shape
  x = y
  for i in reversed(range(0,size)):
    for j in reversed(range(i+1,size)):
      x[i] -= u[i][j]*x[j]
    x[i] /= u[i][i]
  return x

#------------------------------------
# test code
#------------------------------------
#tmp1 = np.array([[2,5,7,12],[4,13,20,14],[8,29,50,16],[3,4,5,6]], dtype=complex)
#tmp2 = np.array([[2,5,7,12],[4,13,20,14],[8,29,50,16],[3,4,5,6]], dtype=complex)
#(l,u) = lu(tmp1)
#print(l.dot(u) - tmp2)

#
#print(l_step(np.array([[1,0,0],[3,1,0],[2,7,2]], dtype=complex), np.array([10,5,10],dtype=complex)))
#print(u_step(np.array([[2,7,2],[0,1,3],[0,0,1]], dtype=complex), np.array([10,5,10],dtype=complex)))
