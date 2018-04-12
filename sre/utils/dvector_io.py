#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import sys
import os

#  sys.path.append('/home/day9011/tensorflow/tensorflow-demo/sre/utils')
#  import kaldi_io

if __name__ == '__main__':
    import kaldi_io
else:
    from . import kaldi_io

os.environ['KALDI_ROOT'] = '/home/day9011/Documents/kaldi'

def write_mat(filename, temp_dict):
    try:
        with open(filename, 'wb') as f:
            for key,mat in temp_dict.iteritems():
                kaldi_io.write_mat(f, mat, key)
        return True
    except Exception, e:
        print(str(e))
        return False


def write_vec(filename, temp_dict):
    try:
        with open(filename, 'wb') as f:
            for key,vec in temp_dict.iteritems():
                kaldi_io.write_vec_flt(f, vec, key)
        return True
    except Exception, e:
        print(str(e))
        return False

def write_to_kaldi_prepare(dirpath, temp_dict, suffix):
    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        utt2spk_file = dirpath.rstrip('/') + '/' + suffix + '-utt2spk'
        with open(utt2spk_file, 'w') as f:
            for key in temp_dict.keys():
                spk_id = key.strip().split('_')[0]
                f.write(key + ' ' + spk_id + '\n')
        return True
    except Exception, e:
        print(str(e))
        return False


def read_mat(filename):
    try:
        temp_dict = {}
        for key,mat in kaldi_io.read_mat_ark(filename):
            if temp_dict.has_key(key):
                raise Exception("duplicated key")
            else:
                temp_dict[key] = mat
        return temp_dict
    except Exception, e:
        print(str(e))
        return False

def read_vec(filename):
    try:
        temp_dict = {}
        for key,vec in kaldi_io.read_vec_flt_ark(filename):
            if temp_dict.has_key(key):
                raise Exception("duplicated key")
            else:
                temp_dict[key] = vec
        return temp_dict
    except Exception, e:
        print(str(e))
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " <filename:string> <type:int(0|1 => Vector|Matrix)>")
        exit(1)
    filename = sys.argv[1]
    filetype = int(sys.argv[2])
    if filetype == 0:
        print(read_mat(filename))
    if filetype == 1:
        vecs = read_vec(filename)
        print(vecs)
        print(vecs[vecs.keys()[0]].shape[0])
