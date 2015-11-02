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
import mylib
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
    a_subsingulars = mylib.singular(a)
    for seq in rdft.all_leading_sequence(size):
      fra_subsingulars.append(mylib.complex_singular(fra[np.ix_(seq,seq)]))
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
  while counter < 5:
    [cd, mca_flg, fs, a, mca, mcfra, fras, a_subcond, fra_subconds, a_subsing, fra_subsing] = const_a_detection(sample, size, rand_range)
    if mca_flg == 1:
      continue
    small_singulars = []
    for sings in fra_subsing:
      if sings.size == 2:
        small_singulars.append(sings[1])
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
      output = open("./result/result" + str(counter) + "_small_singular.txt",'w')
      output.write(str(small_singulars))
      output.close()
      output = open("./result/result" + str(counter) + "_mca.txt",'w')
      output.write(str(mca))
      output = open("./result/result" + str(counter) + "_mcfra.txt",'w')
      output.write(str(mcfra))
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
      output = open("./result/result" + str(counter) + "_graph.txt",'w')
      for i in range(0,len(small_singulars)):
        output.write(str(small_singulars[i]) + " " + str(mcfra[i]) + "\n")
      output.close()
  print("finished")

#------------------------------------
# test code
#------------------------------------
# test_num, sample, size, rand_range
const_a_run(100,100,3,10)
#const_a_test(100,100,3,10,1,[(1,2)])
