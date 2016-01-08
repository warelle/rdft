# coding: UTF-8

import math
import cmath
import random

import scipy.linalg as slinalg
import numpy.linalg as linalg
import numpy as np

import rdft
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

def iteration(a, l, u, b, x, a_cond, rm=[]):
  (size,_) = a.shape
  itera, itermax = 0, 1
  a_nrm = linalg.norm(a, float("inf"))
  x_nrm = linalg.norm(x, float("inf"))
  #cte   = math.log10(a_cond) * math.sqrt(size)/10
  r     = b - a.dot(x)
  r_nrm = linalg.norm(r, float("inf"))
  while itera < itermax:
    y = lu.l_step(l, r)
    z = lu.u_step(u, y)
    if rm == []:
      x = x + z
    else:
      x = x + rm.dot(z)
    r = b - a.dot(x)
    x_nrm = linalg.norm(x, float("inf"))
    r_nrm = linalg.norm(r, float("inf"))
    itera = itera + 1
  return x

def iteration_another(a, l, u, fr, b, x, a_cond, rm=[]):
  (size,_) = a.shape
  itera, itermax = 0, 1
  a_nrm = linalg.norm(a, float("inf"))
  x_nrm = linalg.norm(x, float("inf"))
  #cte   = math.log10(a_cond) * math.sqrt(size)/10
  r     = fr.dot(b - a.dot(x))
  r_nrm = linalg.norm(r, float("inf"))
  while itera < itermax:
    y = lu.l_step(l, r)
    z = lu.u_step(u, y)
    if rm == []:
      x = x + z
    else:
      x = x + rm.dot(z)
    r = fr.dot(b - a.dot(x))
    x_nrm = linalg.norm(x, float("inf"))
    r_nrm = linalg.norm(r, float("inf"))
    itera = itera + 1
  return x


def iteration_step_result(a, l, u, b, x, a_cond):
  (size,_) = a.shape
  itera, itermax = 0, 1
  a_nrm = linalg.norm(a, float("inf"))
  x_nrm = linalg.norm(x, float("inf"))
  x_step_result = [x]
  cte   = math.log10(a_cond) * math.sqrt(size)/10
  r     = b - a.dot(x)
  r_nrm = linalg.norm(r, float("inf"))
  while itera < min(cte, itermax):
    y = lu.l_step(l, r)
    z = lu.u_step(u, y)
    x = x + z
    x_step_result.append(x)
    r = b - a.dot(x)
    x_nrm = linalg.norm(x, float("inf"))
    r_nrm = linalg.norm(r, float("inf"))
    itera = itera + 1
  return x_step_result

#------------------------------------
# test code
#------------------------------------
