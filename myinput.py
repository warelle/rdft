# coding: UTF-8

import math
import cmath
import random
import os

import scipy.linalg as slinalg
import numpy.linalg as linalg
import numpy as np

import rdft
import lib_lu_solve as lib
import iteration
import partial_pivot as pp

import main

#------------------------------------
# function definition
#------------------------------------
def solve_getinfo(a,b,x,sample):
  mat = np.array(a)
  (size,_) = a.shape
  result = []
  for i in range(0, sample):
    f   = rdft.generate_f(size)
    r   = rdft.generate_r(size)
    fr  = f.dot(r)
    ra  = r.dot(a)
    (x1, l, u, fra, frb) = rdft.rdft_lu_solver_with_lu(a,b,r)
    x1  = iteration.iteration(fra, l, u, frb, x1, linalg.cond(fra))
    x1  = iteration.remove_imag(x1)
    err = linalg.norm(x1-x)
    (a_maxcond,_,a_subcond)   = rdft.get_leading_maxcond(mat)
    (fra_maxcond,_,fra_subcond) = rdft.get_leading_maxcond(fra)
    a_cond   = linalg.cond(mat)
    fra_cond = linalg.cond(fra)
    result.append([a_maxcond/a_cond, fra_maxcond/fra_cond, fra, a_subcond, fra_subcond, ra, err])
  return result

def get_singular(a,fra,ra):
  (size,_) = a.shape
  fra_subsings = []
  ra_subsings = []
  _,  a_singular, _ = linalg.svd(a)
  for seq in rdft.all_leading_sequence(size):
    _, fra_subsing, _ = linalg.svd(fra[np.ix_(seq,seq)])
    fra_subsings.append(fra_subsing[-1])
  (s1,s2) = ra.shape
  s3 = min(s1,s2)
  nseq = rdft.all_leading_sequence(size)[-1]
  for seq in rdft.all_leading_sequence(s3):
    _, ra_subsing, _ = linalg.svd(ra[np.ix_(nseq,seq)])
    ra_subsings.append(ra_subsing[-1])
  return (a_singular, fra_subsings, ra_subsings)

def my_a_run(a,b,x, test_num):
  counter = 0
  info = solve_getinfo(a,b,x,test_num)
  for i in range(0, test_num):
    counter = counter + 1
    [mca, mcfra, fra, a_subcond, fra_subconds, ra, err] = info[i]
    print mcfra
    #(a_singular, fra_subsings, ra_subsings) = get_singular(a,fra,ra)
    #cd = linalg.cond(a)
    #np.savetxt("./result/result" + str(counter) + "_a.txt",a)
    #np.savetxt("./result/result_" + str(counter) + "_fra.txt", fra)
    #output = open("./result/result" + str(counter) + "_a_subcond.txt",'w')
    #output.write(str(a_subcond))
    #output.close()
    #output = open("./result/result" + str(counter) + "_fra_subcond.txt",'w')
    #output.write(str(fra_subconds))
    #output.close()
    #output = open("./result/result" + str(counter) + "_mca.txt",'w')
    #output.write(str(mca))
    #output = open("./result/result" + str(counter) + "_mcfra.txt",'w')
    #output.write(str(mcfra))
    #output.close()
    #output = open("./result/result" + str(counter) + "_cdnum.txt",'w')
    #output.write(str(linalg.cond(a)))
    #output.close()
    #output = open("./result/result" + str(counter) + "_a_subsing.txt",'w')
    #output.write(str(a_singular))
    #output.close()
    #output = open("./result/result" + str(counter) + "_ra_subsing.txt",'w')
    #output.write(str(ra_subsings))
    #output.close()
    #output = open("./result/result" + str(counter) + "_fra_subsing.txt",'w')
    #output.write(str(fra_subsings))
    #output.close()
    #output = open("./result/result" + str(counter) + "_err.txt",'w')
    #output.write(str(err))
    #output.close()
  print("finished")

def generate_own_system():
  size = 100
  val_range = 100.0
  a = np.zeros((size,size), dtype=np.complex128)
  for i in range(0,size):
    for j in range(0,size):
    #  if i<=j:
        a[i,j] = random.uniform(-val_range,val_range)
    #    a[i,i] = 1.0
  x = np.zeros(size, dtype=np.complex128)
  for i in range(0,size):
    x[i] = random.uniform(-val_range,val_range)
  b = a.dot(x)
  return (a,b,x)

#------------------------------------
# test code
#------------------------------------
def run(test_num):
  (a,b,x) = generate_own_system()
  my_a_run(a,b,x,test_num)


run(100)
