import ctypes
import ctypes.util

import numpy as np

name = ctypes.util.find_library("m")
mlib = ctypes.cdll.LoadLibrary(name)

def ldsin(x):
  sin  = mlib.sinl
  sin.argtypes = [ctypes.c_longdouble]
  sin.restype  = ctypes.c_longdouble
  return np.array([sin(x[0])],dtype=np.float128)

def ldcos(x):
  cos  = mlib.cosl
  cos.argtypes = [ctypes.c_longdouble]
  cos.restype  = ctypes.c_longdouble
  return np.array([cos(x)],dtype=np.float128)
