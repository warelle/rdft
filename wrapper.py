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

import main

#------------------------------------
# function definition
#------------------------------------
def const_a_detection(sample, size, rand_range):
  test_result = main.const_a(sample, size, rand_range)
  condition_satisfy = 0
  failed_satisfy = 0
  mcfras, fras, a_subconds, fra_subconds = [], [], 0.0, []
  for i in range(0,sample):
    [a, mca, mcfra, fra, a_subcond, fra_subcond] = test_result[i]
    if mca < 20:
      continue
    fras.append(fra)
    mcfras.append(mcfra)
    fra_subconds.append(fra_subcond)
    global mca_up
    if mcfra < 1.5:
      condition_satisfy = condition_satisfy + 1
    if mcfra > 10:
      failed_satisfy = failed_satisfy + 1
  return [condition_satisfy, failed_satisfy, a, mca, mcfras, fras, a_subcond, fra_subconds]

#
# cd   : the number of the tests satisfying the detection condition
# mca  : maxcond(A[k|k])/cond(A)
# mcfra: maxcond[FRA[k|k])/cond(FRA)
# cdnum: condition number of the matrix A
#
def const_a_run(test_num, sample, size, rand_range):
  counter = 0
  failed_counter = 0
  for i in range(0, test_num):
    [cd, fs, a, mca, mcfra, fras, a_subcond, fra_subconds] = const_a_detection(sample, size, rand_range)
    if cd > sample*(2/3):
      print(mylib.singular(a.dot(a)))
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
    else:
      if fs > sample/3:
        np.savetxt("./failed/failed" + str(failed_counter) + "_a.txt",a)
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
        failed_counter = failed_counter + 1
  print(str(counter) + "detected", str(failed_counter) + "failed")

#------------------------------------
# test code
#------------------------------------
# test_num, sample, size, rand_range
const_a_run(100,1000,3,10)
