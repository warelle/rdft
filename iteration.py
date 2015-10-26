# coding: UTF-8

import math
import cmath
import random

import scipy.linalg as slinalg
import numpy.linalg as linalg
import numpy as np

import rdft
import mylib
import lu
import lib_lu_solve as lib

#------------------------------------
# function definition
#------------------------------------
def remove_imag(x):
  (size,) = x.shape
  for i in range(0, size):
    x[i] = x[i].real
  return x

def sum_imag(x):
  (size,) = x.shape
  sum = 0
  for i in range(0,size):
    sum = sum + x[i].imag
  return sum

def iteration(a, l, u, b, x, a_cond):
  (size,_) = a.shape
  itera, itermax = 0, 30
  a_nrm = linalg.norm(a, float("inf"))
  x_nrm = linalg.norm(x, float("inf"))
  cte   = math.log10(a_cond) * math.sqrt(size)/10
  r     = b - a.dot(x)
  r_nrm = linalg.norm(r, float("inf"))
  while itera < min(cte, itermax):
    y = lu.l_step(l, r)
    z = lu.u_step(u, y)
    x = x + z
    r = b - a.dot(x)
    x_nrm = linalg.norm(x, float("inf"))
    r_nrm = linalg.norm(r, float("inf"))
    itera = itera + 1
  return x


#------------------------------------
# test code
#------------------------------------
def generate_random_matrix(size, val_range):
  r = np.zeros((size,size))
  for i in range(0,size):
    for j in range(0,size):
      r[i,j] = math.ceil(random.uniform(-val_range,val_range))
  return r
def generate_random_vector(size,val_range):
  r = np.zeros(size)
  for i in range(0,size):
    r[i] = math.ceil(random.uniform(-val_range,val_range))
  return r
def generate_linear_system(size,val_range):
  return (generate_random_matrix(size,val_range), generate_random_vector(size,val_range))

def mychoice():
  size = 3
  a = np.array([[1,2,0],[4,1,1],[5,3,4]])
  b = np.array([4,9,13])
  return (a,b)

def iteration_check():
  for i in range(0,3):
    size = 100
    (a,b) = generate_linear_system(size, 100)
    #(a,b) = mychoice()
    x1 = []
    l,u = [], []
    (x1, l, u, fra, frb) = rdft.rdft_lu_solver_with_lu(a,b)
    x0 = x1
    x1 = iteration(fra, l, u, frb, x1, linalg.cond(a))
    x2 = lib.lu_solver(a,b)
    x3 = lib.direct_solver(a,b)
    x4 = lib.direct_lu_solver(a,b)
    print("cond:", linalg.cond(a))
    print("---error---")
    print("[0] error", linalg.norm(b - a.dot(x0)))
    print("[1] error", linalg.norm(b - a.dot(x1)))
    print("[2] error", linalg.norm(b - a.dot(x2)))
    print("[3] error", linalg.norm(b - a.dot(x3)))
    print("[4] error", linalg.norm(b - a.dot(x4)))


#iteration_check()
