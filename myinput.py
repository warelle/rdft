# coding: UTF-8

import math
import cmath
import random
import os
import sys

import scipy.linalg as slinalg
import numpy.linalg as linalg
import numpy as np

import rdft
import lib_lu_solve as lib
import iteration
import partial_pivot as pp
import lu

import main

import matrix_generator as mg

#------------------------------------
# function definition
#------------------------------------
def solve_rdft(a,b,x):
  (size,_) = a.shape
  f   = rdft.generate_f(size)
  r   = rdft.generate_r(size)
  fr  = f.dot(r)
  ra  = r.dot(a)
  (x1, l, u, fra, frb, _) = rdft.rdft_lu_solver_with_lu(a,b,r)
  x2 = np.array(x1)
  x2  = iteration.iteration(fra, l, u, frb, x2, linalg.cond(fra))
  x2  = iteration.remove_imag(x2)
  return (x2, fra, fr, ra, x1)
def solve_rdft_improved(a,b,x):
  (size,_) = a.shape
  f   = rdft.generate_f(size)
  r   = rdft.generate_random_r(size) ## === care === ##
  fr  = f.dot(r)
  ra  = r.dot(a)
  (x1, l, u, fra, frb, _) = rdft.rdft_lu_solver_with_lu(a,b,r)
  x2 = np.array(x1)
  x2  = iteration.iteration(fra, l, u, frb, x2, linalg.cond(fra))
  x2  = iteration.remove_imag(x2)
  return (x2, fra, fr, ra, x1)
def solve_gauss(a,b,x):
  (size,_) = a.shape
  g   = rdft.generate_g(size)
  ga  = g.dot(a)
  gb  = g.dot(b)
  (l,u) = lu.lu(ga)
  y     = lu.l_step(l,gb)
  x1    = lu.u_step(u,y)
  x2 = np.array(x1)
  x2  = iteration.iteration_another(a, l, u, g, b, x2, linalg.cond(ga))
  x2  = iteration.remove_imag(x2)
  return (x2, ga, a, x1)

def solve_getinfo(a,b,x,sample):
  mat = np.array(a)
  (size,_) = a.shape
  result = []
  a_cond   = linalg.cond(mat)
  (a_maxcond,_,a_subcond)   = rdft.get_leading_maxcond(mat)
  for i in range(0, sample):
    # RDFT solve
    (x1, fra, fr, ra, x1_orig) = solve_rdft(a,b,x)
    (fra_maxcond,_,fra_subcond) = rdft.get_leading_maxcond(fra)
    fra_cond = linalg.cond(fra)
    err_rdft_iter = linalg.norm(x1-x)
    err_rdft = linalg.norm(x1_orig-x)
    # RDFT improved solve
    (x1_imp, fra_imp, fr_imp, ra_imp, x1_imp_orig) = solve_rdft_improved(a,b,x)
    (fra_maxcond_imp,_,fra_subcond_imp) = rdft.get_leading_maxcond(fra)
    fra_cond_imp = linalg.cond(fra_imp)
    err_rdft_imp_iter = linalg.norm(x1_imp-x)
    err_rdft_imp = linalg.norm(x1_imp_orig-x)
    # GAUSS solve
    (x2, ga, _, x2_orig) = solve_gauss(a,b,x)
    (ga_maxcond,_,ga_subcond) = rdft.get_leading_maxcond(fra)
    ga_cond = linalg.cond(ga)
    err_gauss_iter = linalg.norm(x2-x)
    err_gauss = linalg.norm(x2_orig-x)
    # Partial Pivot
    a_float = np.array(a,dtype=np.float64)
    b_float = np.array(b,dtype=np.float64)
    (x3, pl, pu, swapped_a, swapped_b) = pp.solve(a_float,b_float)
    err_pp = linalg.norm(x3-x)
    ##x3_i = iteration.iteration(swapped_a, pl, pu, swapped_b, x3, linalg.cond(a_float))
    #result make
    result.append(([a_maxcond/a_cond, fra_maxcond/fra_cond, fra, a_subcond, fra_subcond, fr, ra, err_rdft_iter, err_rdft,err_rdft_imp_iter,err_rdft_imp],[ga_maxcond/ga_cond, ga, ga_subcond, err_gauss_iter, err_gauss],[err_pp]))
  return result

def get_singular(a,fra):
  (size,_) = a.shape
  a_fra_subsings = []
  _, a_sings, _ = linalg.svd(a)
  i = 0
  for seq in rdft.all_leading_sequence(size):
    _, fra_subsing, _ = linalg.svd(fra[np.ix_(seq,seq)])
    a_fra_subsings.append( (a_sings[i], fra_subsing[-1]) )
    i = i + 1
  return a_fra_subsings

# opt = 0: A_nk
# opt = 1: A_kn
def nk_singular(a,k, opt):
  (size,_) = a.shape
  nseq = rdft.all_leading_sequence(size)[-1]
  seq  = rdft.all_leading_sequence(k)[-1]
  if opt == 0:
    _, sings, _ = linalg.svd(a[np.ix_(nseq,seq)])
  else:
    _, sings, _ = linalg.svd(a[np.ix_(seq,nseq)])
  return sings

def my_a_run(a,b,x, test_num ,opt=0):
  counter = 0
  info = solve_getinfo(a,b,x,test_num)
  xx = linalg.solve(a,b)
  err_lib = linalg.norm(xx-x)
  for i in range(0, test_num):
    counter = counter + 1
    ([mca, mcfra, fra, a_subcond, fra_subconds, fr, ra, err_rdft_iter, err_rdft, err_rdft_imp_iter, err_rdft_imp],[mcga, ga, ga_subcond, err_gauss_iter, err_gauss],[err_pp]) = info[i]
    (size, _) = a.shape
    print(str(2*opt-1) + " " + str(err_rdft) + " " + str(err_rdft_iter) + " " + str(err_rdft_imp) + " " + str(err_rdft_imp_iter) + " " + str(err_gauss) + " " + str(err_gauss_iter) + " " + str(err_pp)+ " " + str(err_lib))
    #if err_pp > 100:
    #  np.savetxt("./result/result" + str(counter) + "_a.txt",a)
    #####f = rdft.generate_f(size)
    ######print get_singular(a,fra)
    #####_, a_sings, _ = linalg.svd(a)
    #####a_fra_sings   = get_singular(a,fra)
    #####maxafra = 0
    #####for k in range(1,size):
    #####  (_,fra_sing) = a_fra_sings[k]
    ######  print("fra_sing", fra_sing)
    ######  print (k, nk_singular(fr,k,1)[-1], nk_singular(a,k,0)[-1], a_sings[-1], fra_sing, a_sings[-1]/fra_sing)
    #####  if maxafra < a_sings[-1]/fra_sing:
    #####    maxafra = a_sings[-1]/fra_sing
    ######print("maxcond/cond", mcfra)
    ######print(maxafra)
    ######print i
    #####print(str(linalg.cond(a)) + " " + str(err_rdft) + " " + str(err_gauss) + " " + str(err_pp))
    #print err_rdft
    #print err_gauss
    #print err_pp
    #print("mcfra/maxsfra:" + str(mcfra) + " " + str(maxafra))
    #if mcfra/maxafra > 1.1:
    #  print("=========================================================================")
    #  break
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
    #output.write(str(err_rdft))
    #output.close()
  #print("finished")

def generate_own_system(size, val_range, opt=0,shift=0):
  #a = mg.random(size, val_range)
  #a = mg.random_well_condition(size, val_range)
  #a = mg.z_matrix(size, 100)
  #a = mg.identity(size)
  #a = mg.lower_triangle(size, val_range)
  a = mg.diag_big(size, val_range*1.0e14, val_range)
  #a = mg.ascending_vector_matrix(size)
  #a = mg.toeplitz(size,100)
  #a = mg.circular(size,100,opt)
  #a = mg.circular_like(size,100,opt)
  #a = mg.circular_shift(size,100,opt,shift)
  #a = mg.circular_partitioned(size,100,opt)
  #a = mg.obi1(size,100,opt)
  #a = mg.obi2(size,100,opt)
  #a = mg.obi3(size,100,opt)
  #a = mg.sparse1(size,100)
  #a = mg.sparse2(size,100,opt)
  #a = mg.sparse3(size,100,opt)
  #a = mg.arrowhead(size,100)
  #a = mg.arrowbottom(size,100)
  #print(linalg.cond(a))
  x = np.zeros(size, dtype=np.complex128)
  for i in range(0,size):
    x[i] = random.uniform(-val_range,val_range)
  b = a.dot(x)
  return (a,b,x)

#------------------------------------
# test code
#------------------------------------
def run(test_num, opt=0,shift=0):
  (a,b,x) = generate_own_system(128, 100.0, opt,shift)
  my_a_run(a,b,x,test_num,opt)


def run_run(run_num,test_num):
  for i in range(1,run_num):
    sys.stderr.write(str(i)+"\n")
    for j in range(0,5):
      run(test_num,i)

run_run(64,1)
