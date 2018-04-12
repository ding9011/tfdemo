#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.io.wavfile as wav
import numpy as np
import speechpy
import tables as tb
import os
import sys
from utils.dvector_io import *
from utils.data_tool import save_label_maps
from utils.config import *
from utils.tools import *
init_config("config/my.conf")
set_section("data")

#  a = tf.constant([1,2,3,4,5,6], shape=[2,2,2], dtype=tf.float64)
#  b = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12], shape=[2,2,3], dtype=tf.float64)
a = tf.constant([1,2,3,4,5,6], shape=[2,3], dtype=tf.float64)
b = tf.constant([4,5,6,7,8,9], shape=[2,3], dtype=tf.float64)
result = tf.divide(tf.add(a,b), 2.0)
#  concat = tf.concat([a,b], -1)
#  product = tf.matmul(a,b)
#  norm = tf.nn.l2_normalize(product, -1)
#  test_shape = tf.identity(a)
test_shape = tf.pad(a, [[1, 1], [0, 0]])
#  initializer = tf.contrib.layers.xavier_initializer()
#  test_shape = tf.get_variable(shape=[3,4], initializer=initializer, name="test")
#  print(test_shape.shape)
#  print(test_shape.get_shape()[-1].value)
#  get_test = test_shape[:, :, 1, :]
#  reshape_test = tf.reshape(get_test, [3,3,3])
#  product = tf.einsum('ijk,jk->ijk',a,b)


if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #  re_a,re_b,result = sess.run([a,b,result])
        re_a,re_b,result, re_test = sess.run([a,b,result, test_shape])
        #  re_a,re_b,result, re_test, re_shape = sess.run([a,b,result, test_shape, shape])
        #  re_a,re_b,result, re_norm, re_concat = sess.run([a,b,product, norm, concat])
        #  print(re_a.shape)
        print(re_a)
        print(re_b)
        print(result)
        print(re_test)
        #  print(re_shape)
        #  print(re_norm)
        #  print(re_concat)
        #  re_a,re_b,result = sess.run([test_shape, get_test, reshape_test])
        #  print("test_shape:",re_a)
        #  print("get one shape:",re_b)
        #  print("reshape:", result)
    #  if len(sys.argv) != 2:
    #      print("Usage: " + sys.argv[0] + " <ark>")
    #      exit(1)
    #  file_name = sys.argv[1]
    #  feats = read_mat(file_name)
    #  print(feats)
    #  print("number of utt:", len(feats.keys()))
