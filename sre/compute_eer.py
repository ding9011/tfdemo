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

def eer(trials, enroll_dict, test_dict=None):
    if test_dict == None:
        eer, eer_th = compute_eer(trials, enroll_dict, enroll_dict)
        print("eer: %.4f, threshold: %.4f" % (eer, eer_th))
    else:
        eer, eer_th = compute_eer(trials, enroll_dict, test_dict)
        print("eer: %.4f, threshold: %.4f" % (eer, eer_th))



if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: " + sys.argv[0] + " <trials:string> <enroll_file:string> [<test_file:string>]")
        exit(1)
    trials = sys.argv[1]
    if len(sys.argv) == 3:
        enroll_file = sys.argv[2]
        enroll_dict = dvector_io.read_vec(enroll_file)
        eer(trials, enroll_dict)
    else:
        enroll_file = sys.argv[2]
        test_file = sys.argv[3]
        enroll_dict = dvector_io.read_vec(enroll_file)
        test_dict = dvector_io.read_vec(test_file)
        eer(trials, enroll_dict, test_dict)
