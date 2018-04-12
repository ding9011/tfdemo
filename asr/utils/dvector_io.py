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
    from KaldiArkIO import *
else:
    from .KaldiArkIO import *

def write_mat(filename, temp_dict):
    try:
        Writer writer(filename)
        for key,mat in temp_dict.items():
            writer.write_next_utt(filename, key, mat)
        writer.close()
        return True
    except Exception(e):
        print(str(e))
        return False


def write_vec(filename, temp_dict):
    try:
        Writer writer(filename)
        for key,vec in temp_dict.items():
            np.reshape(vec, [1, a.shape[0]])
            writer.write_next_utt(filename, key, mat)
        writer.close()
        return True
        with open(filename, 'wb') as f:
            for key,vec in temp_dict.items():
                kaldi_io.write_vec_flt(f, vec, key)
        return True
    except Exception(e):
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
    except Exception(e):
        print(str(e))
        return False


def read_mat(filename):
    try:
        temp_dict = {}
        for key,mat in kaldi_io.read_mat_ark(filename):
            print("key", key)
            print("mat", mat)
            exit(1)
            if key in temp_dict:
                raise Exception("duplicated key")
            else:
                temp_dict[key] = mat
        return temp_dict
    except Exception(e):
        print(str(e))
        return False

def read_vec(filename):
    try:
        temp_dict = {}
        for key,vec in kaldi_io.read_vec_flt_ark(filename):
            if key in temp_dict:
                raise Exception("duplicated key")
            else:
                temp_dict[key] = vec
        return temp_dict
    except Exception(e):
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
