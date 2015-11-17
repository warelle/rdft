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
def const_a_detection(sample, size, rand_range):
  test_result = main.const_a(sample, size, rand_range)
  condition_satisfy = 0
  failed_satisfy = 0
  mcfras, fras, a_subconds, fra_subconds, a_subsingulars, fra_subsingulars = [], [], 0.0, [], 0.0, []
  mca_flg = 0
  for i in range(0,sample):
    [a, mca, mcfra, fra, a_subcond, fra_subcond] = test_result[i]
    fras.append(fra)
    mcfras.append(mcfra)
    _,  a_subsingulars, _ = linalg.svd(a)
    for seq in rdft.all_leading_sequence(size):
      _, fra_subsing, _ = linalg.svd(fra[np.ix_(seq,seq)])
      fra_subsingulars.append(fra_subsing)
    fra_subconds.append(fra_subcond)
    if mca < 20:
      mca_flg = 1 # this is not what we want to detect even as a sample
      continue
    if mcfra < 1.5:
      condition_satisfy = condition_satisfy + 1
    if mcfra > 10:
      failed_satisfy = failed_satisfy + 1
  return [condition_satisfy, mca_flg, failed_satisfy, a, mca, mcfras, fras, a_subcond, fra_subconds, a_subsingulars, fra_subsingulars]

#
# cd   : the number of the tests satisfying the detection condition
# mca  : maxcond(A[k|k])/cond(A)
# mcfra: maxcond[FRA[k|k])/cond(FRA)
# cdnum: condition number of the matrix A
#
def const_a_run(test_num, sample, size, rand_range):
  counter = 0
  failed_counter = 0
  sample_counter = 0
  #for i in range(0, test_num):
  while counter < 100:
    [cd, mca_flg, fs, a, mca, mcfra, fras, a_subcond, fra_subconds, a_subsing, fra_subsing] = const_a_detection(sample, size, rand_range)
    if mca_flg == 1:
      continue
    if cd > sample*(2/3):
      counter = counter + 1
      np.savetxt("./result/result" + str(counter) + "_a.txt",a)
      os.mkdir("./result/fra_" + str(counter))
      for i in range(0,len(fras)):
        np.savetxt("./result/fra_" + str(counter) + "/" + str(i) + ".txt", fras[i])
      output = open("./result/result" + str(counter) + "_a_subcond.txt",'w')
      output.write(str(a_subcond))
      output.close()
      output = open("./result/result" + str(counter) + "_fra_subcond.txt",'w')
      output.write(str(fra_subconds))
      output.close()
      output = open("./result/result" + str(counter) + "_mca.txt",'w')
      output.write(str(mca))
      output = open("./result/result" + str(counter) + "_mcfra.txt",'w')
      output.write(str(mcfra))
      output.close()
      output = open("./result/result" + str(counter) + "_det.txt",'w')
      output.write(str(np.linalg.det(a)))
      output.close()
      output = open("./result/result" + str(counter) + "_cd.txt",'w')
      output.write(str(cd))
      output.close()
      output = open("./result/result" + str(counter) + "_cdnum.txt",'w')
      output.write(str(linalg.cond(a)))
      output.close()
      output = open("./result/result" + str(counter) + "_a_subsing.txt",'w')
      output.write(str(a_subsing))
      output.close()
      output = open("./result/result" + str(counter) + "_fra_subsing.txt",'w')
      output.write(str(fra_subsing))
      output.close()
      print(counter)
    elif fs > sample/3:
      np.savetxt("./failed/failed" + str(failed_counter) + "_a.txt",a)
      os.mkdir("./failed/fra_" + str(faled_counter))
      for i in range(0,len(fras)):
        np.savetxt("./failed/fra_" + str(failed_counter) + "/" + str(i) + ".txt", fras[i])
      output = open("./failed/result" + str(failed_counter) + "_a_subcond.txt",'w')
      output.write(str(a_subcond))
      output.close()
      output = open("./failed/result" + str(failed_counter) + "_fra_subcond.txt",'w')
      output.write(str(fra_subconds))
      output.close()
      output = open("./failed/failed" + str(failed_counter) + "_mca.txt",'w')
      output.write(str(mca))
      output = open("./failed/failed" + str(failed_counter) + "_mcfra.txt",'w')
      output.write(str(mcfra))
      output.close()
      output = open("./failed/failed" + str(failed_counter) + "_det.txt",'w')
      output.write(str(np.linalg.det(a)))
      output.close()
      output = open("./failed/failed" + str(failed_counter) + "_cd.txt",'w')
      output.write(str(cd))
      output.close()
      output = open("./failed/failed" + str(failed_counter) + "_cdnum.txt",'w')
      output.write(str(linalg.cond(a)))
      output.close()
      output = open("./failed/failed" + str(counter) + "_a_subsing.txt",'w')
      output.write(str(a_subsing))
      output.close()
      output = open("./failed/failed" + str(counter) + "_fra_subsing.txt",'w')
      output.write(str(fra_subsing))
      output.close()
      failed_counter = failed_counter + 1
    elif sample_counter < 10:
      np.savetxt("./sample/sample" + str(sample_counter) + "_a.txt",a)
      os.mkdir("./sample/fra_" + str(sample_counter))
      for i in range(0,len(fras)):
        np.savetxt("./sample/fra_" + str(sample_counter) + "/" + str(i) + ".txt", fras[i])
      output = open("./sample/sample" + str(sample_counter) + "_a_subcond.txt",'w')
      output.write(str(a_subcond))
      output.close()
      output = open("./sample/sample" + str(sample_counter) + "_fra_subcond.txt",'w')
      output.write(str(fra_subconds))
      output.close()
      output = open("./sample/sample" + str(sample_counter) + "_mca.txt",'w')
      output.write(str(mca))
      output = open("./sample/sample" + str(sample_counter) + "_mcfra.txt",'w')
      output.write(str(mcfra))
      output.close()
      output = open("./sample/sample" + str(sample_counter) + "_det.txt",'w')
      output.write(str(np.linalg.det(a)))
      output.close()
      output = open("./sample/sample" + str(sample_counter) + "_cd.txt",'w')
      output.write(str(cd))
      output.close()
      output = open("./sample/sample" + str(sample_counter) + "_cdnum.txt",'w')
      output.write(str(linalg.cond(a)))
      output.close()
      output = open("./sample/sample" + str(counter) + "_a_subsing.txt",'w')
      output.write(str(a_subsing))
      output.close()
      output = open("./sample/sample" + str(counter) + "_fra_subsing.txt",'w')
      output.write(str(fra_subsing))
      output.close()
      sample_counter = sample_counter + 1
  print(str(counter) + "detected", str(failed_counter) + "failed", str(sample_counter) + "samples")


# way: -1:none, 0:transpose A, 1:swap column
def const_a_test(test_num, sample, size, rand_range, way=-1, way_option=[]):
  counter = 0
  sample_counter = 0
  for i in range(0, test_num):
    [cd, fs, mat, mca, mcfra, fras, a_subcond, fra_subconds] = const_a_detection(sample, size, rand_range)
    a   = np.array(mat)
    if way == 0:
      mat = mat.T
    elif way == 1:
      for (k,j) in way_option:
        mat = pp.swap(mat,k,j)
    if cd > sample*(2/3):
      np.savetxt("./test_result/test_result" + str(counter) + "_a.txt",a)
      os.mkdir("./test_result/fra_" + str(counter))
      for i in range(0,len(fras)):
        np.savetxt("./test_result/fra_" + str(counter) + "/" + str(i) + ".txt", fras[i])
      output = open("./test_result/test_result" + str(counter) + "_a_subcond.txt",'w')
      output.write(str(a_subcond))
      output.close()
      output = open("./test_result/test_result" + str(counter) + "_fra_subcond.txt",'w')
      output.write(str(fra_subconds))
      output.close()
      output = open("./test_result/test_result" + str(counter) + "_mca.txt",'w')
      output.write(str(mca))
      output = open("./test_result/test_result" + str(counter) + "_mcfra.txt",'w')
      output.write(str(mcfra))
      output.close()
      output = open("./test_result/test_result" + str(counter) + "_det.txt",'w')
      output.write(str(np.linalg.det(a)))
      output.close()
      output = open("./test_result/test_result" + str(counter) + "_cd.txt",'w')
      output.write(str(cd))
      output.close()
      output = open("./test_result/test_result" + str(counter) + "_cdnum.txt",'w')
      output.write(str(linalg.cond(a)))
      output.close()
      sample_counter = 0
      a_subcond, fra_subconds = [], []
      while sample_counter < 100:
        f   = rdft.generate_f(size)
        r   = rdft.generate_r(size)
        fr  = f.dot(r)
        fra = fr.dot(mat)
        (a_maxcond,_,a_subcond)   = rdft.get_leading_maxcond(mat)
        (fra_maxcond,_,fra_subcond) = rdft.get_leading_maxcond(fra)
        fra_subconds.append(fra_subcond)
        sample_counter = sample_counter + 1
      np.savetxt("./test_result/test_result" + str(counter) + "_z_mod_a.txt",mat)
      os.mkdir("./test_result/fra_" + str(counter) + "_" + str(sample_counter))
      for i in range(0,len(fras)):
        np.savetxt("./test_result/fra_" + str(counter) + "_" + str(sample_counter) + "/" + str(i) + ".txt", fras[i])
      output = open("./test_result/test_result" + str(counter) + "_z_a_subcond.txt",'w')
      output.write(str(a_subcond))
      output.close()
      output = open("./test_result/test_result" + str(counter) + "_z_fra_subcond.txt",'w')
      output.write(str(fra_subconds))
      output.close()
      output = open("./test_result/test_result" + str(counter) + "_z_mca.txt",'w')
      output.write(str(mca))
      output = open("./test_result/test_result" + str(counter) + "_z_mcfra.txt",'w')
      output.write(str(mcfra))
      output.close()
      output = open("./test_result/test_result" + str(counter) + "_z_det.txt",'w')
      output.write(str(np.linalg.det(mat)))
      output.close()
      output = open("./test_result/test_result" + str(counter) + "_z_cdnum.txt",'w')
      output.write(str(linalg.cond(mat)))
      output.close()
      counter = counter + 1
  print(str(counter) + "detected")

#------------------------------------
# test code
#------------------------------------
# test_num, sample, size, rand_range
const_a_run(100,100,3,10)
#const_a_test(100,100,3,10,1,[(1,2)])
