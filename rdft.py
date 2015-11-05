# coding: UTF-8

import math
import cmath
import random

import scipy.linalg as slinalg
import numpy.linalg as linalg
import numpy as np

import lu
import mylib

#------------------------------------
# function definition
#------------------------------------
def generate_f(size):
  #return mylib.generate_f_m(size)
  r = np.zeros((size,size), dtype=np.complex256)
  for i in range(0,size):
    for j in range(0,size):
      r[i,j] = (cmath.rect(1,(-2.0*math.pi/size)*i*j))/cmath.sqrt(size)
      #r[i,j] = (mylib.cos_float128(-2.0*math.pi*i*j/size) + 1j*mylib.sin_float128(-2.0*math.pi*i*j/size))/cmath.sqrt(size)
  return np.array(r,dtype=np.complex256)

def generate_r(size):
  r = np.zeros((size, size), dtype=np.complex256)
  for i in range(0,size):
    randnum = random.uniform(0, 2*math.pi)
    r[i,i]  = cmath.rect(1, randnum)
  return r

def all_leading_sequence(maxsize):
  r = []
  for i in range(0,maxsize):
    als = []
    for j in range(0,i+1):
      als.append(j)
    r.append(als)
  return r

def get_leading_maxcond(mat):
  (size,_)   = mat.shape
  allseq = all_leading_sequence(size)
  maxcond, maxcond_seq = -1, -1
  leading_cond = []
  for seq in allseq:
    sub = mat[np.ix_(seq,seq)]
    subcond = mylib.cond(sub)
    leading_cond.append(subcond)
    if subcond > maxcond:
      maxcond     = subcond
      maxcond_seq = seq
  return (maxcond, maxcond_seq, leading_cond)

def rdft(a, r=[]):
  (size,_) = a.shape
  f = generate_f(size)
  if r == []:
    r = generate_r(size)
  fr  = f.dot(r)
  fra = fr.dot(a)
  return (fra, fr)

def rdft_lu_solver(a,b,r=[]):
  (fra, fr) = rdft(a,r)
  frb       = fr.dot(b)
  (l,u) = lu.lu(fra)
  y     = lu.l_step(l,frb)
  x     = lu.u_step(u,y)
  return x

def rdft_lu_solver_with_lu(a,b,r=[]):
  (fra, fr) = rdft(a,r)
  frb       = fr.dot(b)
  fra_save  = np.array(fra)
  frb_save  = np.array(frb)
  (l,u) = lu.lu(fra)
  y     = lu.l_step(l,frb)
  x     = lu.u_step(u,y)
  return (x, l, u, fra_save, frb_save, fr)

#------------------------------------
# test code
#------------------------------------
#for i in range(0, 1):
#  size = 300
#  mat = np.zeros((size,size))
#  for j in range(0,size):
#    for k in range(0,size):
#      mat[j,k] = random.uniform(-100,100)
#  f = generate_f(size)
#  r = generate_r(size)
#  fr  = f.dot(r)
#  fra = fr.dot(mat)
#  (a_maxcond,_)   = get_leading_maxcond(mat)
#  (fra_maxcond,_) = get_leading_maxcond(fra)
#  a_cond   = linalg.cond(mat)
#  fra_cond = linalg.cond(fra)
#  print("A:  ", a_maxcond/a_cond)
#  print("FRA:", fra_maxcond/fra_cond)
#
