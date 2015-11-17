# coding: UTF-8

import math
import cmath
import random

import scipy.linalg as slinalg
import numpy.linalg as linalg
import numpy as np

import rdft
import lib_lu_solve as lib
import iteration
import partial_pivot as pp

#------------------------------------
# function definition
#------------------------------------
def generate_random_matrix(size, val_range):
  r = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
      r[i,j] = random.uniform(-val_range,val_range)
  return r
def generate_random_vector(size,val_range):
  r = np.zeros(size, dtype=np.complex128)
  for i in range(0,size):
    r[i] = random.uniform(-val_range,val_range)
  return r
def generate_linear_system(size,val_range):
  a = generate_random_matrix(size,val_range)
  x = generate_random_vector(size, val_range)
  b = a.dot(x)
  return (a, b, x)

def error(x1, x2):
  dv = x1 - x2
  d  = linalg.norm(dv)
  return d

#------------------------------------
# test code
#------------------------------------
def const_r():
  for i in range(0, 50):
    size = 200
    mat = np.zeros((size,size))
    r   = rdft.generate_r(size)            # use constant r
    for j in range(0,size):
      for k in range(0,size):
        mat[j,k] = random.uniform(-100,100)
    f = rdft.generate_f(size)
    fr  = f.dot(r)
    fra = fr.dot(mat)
    (a_maxcond,_,_)   = rdft.get_leading_maxcond(mat)
    (fra_maxcond,_,_) = rdft.get_leading_maxcond(fra)
    a_cond   = linalg.cond(mat)
    fra_cond = linalg.cond(fra)
    print("A:  ", a_maxcond/a_cond)
    print("FRA:", fra_maxcond/fra_cond)

def const_a(sample, size, rand_range):
  result = []
  mat = np.zeros((size,size))
  for j in range(0,size):
    for k in range(0,size):
      mat[j,k] = random.uniform(-rand_range,rand_range)
  for i in range(0, sample):
    f   = rdft.generate_f(size)
    r   = rdft.generate_r(size)
    fr  = f.dot(r)
    fra = fr.dot(mat)
    (a_maxcond,_,a_subcond)   = rdft.get_leading_maxcond(mat)
    (fra_maxcond,_,fra_subcond) = rdft.get_leading_maxcond(fra)
    a_cond   = linalg.cond(mat)
    fra_cond = linalg.cond(fra)
    result.append([mat, a_maxcond/a_cond, fra_maxcond/fra_cond, fra, a_subcond, fra_subcond])
    #print("A:  ", a_maxcond/a_cond)
    #print("FRA:", fra_maxcond/fra_cond)
  return result

def fourier_r(size):
  r = np.zeros((size, size), dtype=complex)
  for i in range(0,size):
    r[i,i] = (cmath.rect(1,(-2.0*math.pi/size)*i*i))
  return r


def iteration_checker(size, test_num, val_range, res_opt=0):
  for i in range(0,test_num):
    (a,b,x) = generate_linear_system(size, val_range)
    a_float = np.array(a,dtype=np.float64)
    b_float = np.array(b,dtype=np.float64)
    x1 = []
    l,u = [], []
    (x1, l, u, fra, frb) = rdft.rdft_lu_solver_with_lu(a,b)
    x0 = x1
    x1_after  = np.array(x1)
    x1_after  = iteration.iteration(fra, l, u, frb, x1_after, linalg.cond(fra))
    x1_after  = iteration.remove_imag(x1_after)
    (x2, pl, pu, swapped_a, swapped_b) = pp.solve(a_float,b_float)
    x3 = iteration.iteration(swapped_a, pl, pu, swapped_b, x2, linalg.cond(a_float))
    #x2 = lib.lu_solver(a,b)
    #x2 = lib.direct_solver(a_float,b_float)
    #x4 = lib.direct_lu_solver(a,b)
    if res_opt == 1:
      print("---error---")
      print("cond:", linalg.cond(a))
      print("[0] error", linalg.norm(b - a.dot(x0)))
      print("[1] error", linalg.norm(b - a.dot(x1)))
      #print("[2] error", linalg.norm(b - a.dot(x2)))
      #print("[3] error", linalg.norm(b - a.dot(x3)))
    elif res_opt == 2:
      print(str(linalg.cond(a)) + " " +
      str(linalg.norm(x - x0)) + " " +
      str(linalg.norm(x - x1_before)) + " " +
      str(linalg.norm(x - x1_after)) + " " +
      str(linalg.norm(x - x2)) + " " +
      str(linalg.norm(x - x3)))
    elif res_opt != 1:
      print("---error---")
      print("cond:", linalg.cond(a))
      print("[0] error", linalg.norm(x - x0))
      print("[1] error", linalg.norm(x - x1_before))  # remove_imag => iteration
      print("[2] error", linalg.norm(x - x1_after))   # iteration   => remove_imag
      print("[3] error", linalg.norm(x - x2))
      #print("[4] error", linalg.norm(x - x3))
    #print(x1)
    # print("[4] error", linalg.norm(b - a.dot(x4))) # no need because the same result as [3]



#error_check(True)

# size, test_num, val_range, graph
#iteration_checker(100, 100, 100, 2)
