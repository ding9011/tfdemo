#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tables as tb
import numpy as np
from utils.dvector_tool import compute_eer
import os
import utils.dvector_io as dvector_io
import sys
import model.score_model as sc_model

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " <train_dvector:string>" + " <number of vector:int>")
        exit(1)
    train_file = sys.argv[1]
    k = int(sys.argv[2])
    train_dict = dvector_io.read_vec(train_file)
    P = sc_model.matrix_eig(train_dict, k)
    np.save("eig.mat", P)
