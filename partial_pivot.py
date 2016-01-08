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
def swap(a,i,j):
  ai = np.array(a[i,:])
  a[i,:] = np.array(a[j,:])
  a[j,:] = np.array(ai)
  return a

def pivot(a, k):
  (size,_) = a.shape
  pivoter, p = -1, k
  for i in range(k,size):
    if math.fabs(a[i,k]) > pivoter:
      pivoter = math.fabs(a[i,k])
      p       = i
  if p != k:
    a = swap(a,k,p)
  return p

def lu(mat):
  (size, _) = mat.shape
  l, u = np.zeros((size,size), dtype=np.float64), np.zeros((size,size), dtype=np.float64)
  p = []
  for i in range(0,size-1):
    pnum = pivot(mat,i)
    p.append(pnum)
    print p
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
  return (l,u,p)

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

def swap_a(a,p):
  (size,_) = a.shape
  for i in range(0, size-1):
    ai = np.array(a[i,:])
    a[i,:] = np.array(a[p[i],:])
    a[p[i],:] = np.array(ai)
  return a

def swap_b(b,p):
  (size,) = b.shape
  for i in range(0,size-1):
    tmp = b[i]
    b[i] = b[p[i]]
    b[p[i]] = tmp
  return b

def solve(a,b):
  a_save = np.array(a)
  b_save = np.array(b)
  (l,u,p) = lu(a)
  swapped_b = swap_b(b,p)
  swapped_b_save = np.array(swapped_b)
  tmp1 = l_step(l, swapped_b)
  x = u_step(u, tmp1)
  swapped_a = swap_a(a_save,p)
  return (x, l, u, swapped_a, swapped_b_save)

def solve_cast(a,b):
  a_save = np.array(a)
  b_save = np.array(b)
#  a = mylib.float128_64_128(a)
#  b = mylib.float128_64_128(b)
  (l,u,p) = lu(a)
  swapped_b = swap_b(b,p)
  swapped_b_save = np.array(swapped_b)
#  l = mylib.float128_64_128(l)
#  u = mylib.float128_64_128(u)
  tmp1 = l_step(l, swapped_b)
  x = u_step(u, tmp1)
#  x = mylib.float128_64_128(x)
  swapped_a = swap_a(a_save,p)
  return (x, l, u, swapped_a, swapped_b_save)

#------------------------------------
# test code
#------------------------------------
#tmp1 = np.array([[2,5,7],[4,13,20],[8,29,50]], dtype=np.float64)
#tmp2 = np.array([14,37,87], dtype=np.float64)
#print(tmp1)
#print(tmp2)
#(l,u,p) = lu(tmp1)
#print(l)
#print(u)
#print(p)
#tmp3 = swap_b(tmp2,p)
#print(tmp3,p)
#print("L*U", l.dot(u))
#tmp4 = l_step(l,tmp3)
#tmp5 = u_step(u,tmp4)
#print(tmp5)



