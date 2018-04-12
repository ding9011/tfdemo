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
from utils.data_tool import produce_utt_samples
from utils.dvector_io import *
from utils.config import *
from utils.tools import *
init_config("config/my.conf")

def make_egs(_feats_scp, _egs_dir):
    set_section("sample")
    num_samples_per_ark = int(get("num_samples_per_ark"))
    window_size = int(get("window_size"))
    window_shift = int(get("window_shift"))
    set_section("train")
    left_context = int(get("left_context"))
    right_context = int(get("right_context"))
    if os.path.exists(_egs_dir):
        empty_dir(_egs_dir)
    if not os.path.exists(_egs_dir):
        os.makedirs(_egs_dir)
    ark_path = os.path.join(os.path.abspath(_egs_dir), 'egs_arks')
    if os.path.exists(ark_path):
        empty_dir(ark_path)
    if not os.path.exists(ark_path):
        os.makedirs(ark_path)
    egs_scp = os.path.join(os.path.abspath(_egs_dir), 'egs.scp')
    total_samples = os.path.join(os.path.abspath(_egs_dir), 'total_samples')
    if os.path.exists(egs_scp):
        os.remove(egs_scp)
    feats_scp_lines = open(_feats_scp, 'r').readlines()
    ark_index = 1
    num_samples = 0
    sample_dict = {}
    for line in feats_scp_lines:
        feats = read_mat(line.strip())
        for k,v in feats.iteritems():
            feat = v
            while feat.shape[0] < window_size:
                feat = np.row_stack((feat, v))
            samples = produce_utt_samples(feat, window_size, window_shift, left_context, right_context)
            for i in range(len(samples)):
                sample_id = k + '-' + str(i)
                sample_dict[sample_id] = samples[i]
                num_samples += 1
                if num_samples % num_samples_per_ark == 0:
                    egs_ark_file = os.path.join(os.path.abspath(ark_path), 'egs.%d.ark' % (ark_index))
                    write_mat(egs_ark_file, sample_dict)
                    with open(egs_scp, 'a') as f:
                        f.write(egs_ark_file + '\n')
                        f.close()
                    print("store %d samples into %s" % (len(sample_dict.keys()), egs_ark_file))
                    sample_dict = {}
                    ark_index += 1
    if not sample_dict:
        egs_ark_file = os.path.join(os.path.abspath(ark_path), 'egs.%d.ark' % (ark_index))
        write_mat(egs_ark_file, sample_dict)
        print("store %d samples into %s") % (len(sample_dict.keys()), egs_ark_file)
    print("finish handle %d samples" % (num_samples))
    with open(total_samples, 'w') as f:
        f.write(str(num_samples))
        f.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " <string:feats.scp> <string:out_dirname>")
        exit(1)
    feats_scp = sys.argv[1]
    egs_dir = sys.argv[2]
    make_egs(feats_scp, egs_dir)
