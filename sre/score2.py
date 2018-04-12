#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tables as tb
import numpy as np
from utils.dvector_tool import *
import os
import utils.dvector_io as dvector_io
import sys
from sklearn.externals import joblib


def eer(trials, enroll_dict, test_dict=None):
    if test_dict == None:
        eer, eer_th = compute_eer(trials, enroll_dict, enroll_dict)
        print("eer: %.4f, threshold: %.4f" % (eer, eer_th))
    else:
        eer, eer_th = compute_eer(trials, enroll_dict, test_dict)
        print("eer: %.4f, threshold: %.4f" % (eer, eer_th))



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " <trials:string> <test-dvector:string> <lda_model:string>")
        exit(1)
    trials = sys.argv[1]
    test_dvector = sys.argv[2]
    lda_model_file = sys.argv[3]
    test_dict = dvector_io.read_vec(test_dvector)
    clf = joblib.load(lda_model_file)
    origin_vecs = []
    for _, vec in test_dict.iteritems():
        origin_vecs.append(vec)
    transformed_vecs = clf.transform(origin_vecs)
    print("transformed_vecs shape:", transformed_vecs.shape)
    i = 0
    for key, vec in test_dict.iteritems():
        test_dict[key] = transformed_vecs[i]
        i += 1
    eer(trials, test_dict)
