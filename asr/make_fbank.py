#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.io.wavfile as wav
import numpy as np
import speechpy
import os
import sys
from utils.dvector_io import *
from utils.data_tool import save_label_maps
from utils.config import *
from utils.tools import *
init_config("config/my.conf")
set_section("data")
sample_rate = int(get('sample_rate'))
frame_length = float(get('frame_length'))
frame_stride = float(get('frame_stride'))
low_frequency = int(get('low_frequency'))
high_frequency = int(get('high_frequency'))
num_utt_per_ark = int(get('num_utt_per_ark'))

set_section("train")
feat_dim = int(get("feature_dim"))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " <string:wav.scp> <string:out_dirname>")
        exit(1)
    file_name = sys.argv[1]
    data_dir = sys.argv[2]
    feats_dir = os.path.join(os.path.abspath(data_dir), 'feats')
    feats_file = os.path.join(os.path.abspath(data_dir), 'feats.scp')
    num_frames_file = os.path.join(os.path.abspath(data_dir), 'num_frames')
    map_path = os.path.join(os.path.abspath(data_dir), 'label_map')
    if os.path.exists(map_path):
        os.remove(map_path)
    if os.path.exists(feats_file):
        os.remove(feats_file)
    if os.path.exists(num_frames_file):
        os.remove(num_frames_file)
    if not os.path.exists(feats_dir):
        os.makedirs(feats_dir)
    else:
        empty_dir(feats_dir)
        os.makedirs(feats_dir)
    wav_scp = open(file_name, 'r').readlines()
    num_wav = 0
    c_spk_id = 0
    label_map = {}
    feats_dict = {}
    ark_index = 1
    for line in wav_scp:
        wav_info = line.strip().split(' ')
        utt_id = wav_info[0]
        spk_id = utt_id.strip().split('_')[0]
        utt_file = wav_info[1].strip()
        fs, signal = wav.read(utt_file)
        if int(fs) != int(sample_rate):
            print("WARNING: file %s has error sample_rate %d" % (utt_file, int(fs)))
            continue
        lmfe = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=frame_length,
            frame_stride=frame_stride, num_filters=feat_dim, fft_length=512, low_frequency=low_frequency, high_frequency=high_frequency)
        if lmfe.shape[0] < 50:
            print("WARNING:the [%s] wav file is no use" % (spk_id))
            continue
        if not spk_id in label_map.keys():
            label_map[spk_id] = c_spk_id
            c_spk_id += 1
        with open(num_frames_file, 'a') as f:
            f.write(utt_id + ' ' + str(lmfe.shape[0]) + '\n')
            f.close()
        feats_dict[utt_id] = lmfe
        print("finish %s[%d] wav %s " % (spk_id, label_map[spk_id], utt_id), ",utterance shape:", lmfe.shape)
        num_wav += 1
        if num_wav % num_utt_per_ark == 0:
            feats_path = os.path.join(os.path.abspath(feats_dir), 'fbank.%d.ark' % (ark_index))
            with open(feats_file, 'a') as f:
                f.write(feats_path + '\n')
                f.close()
            write_mat(feats_path, feats_dict)
            feats_dict = {}
            print("store %d utt fbank feats into %s" % (len(feats_dict), feats_path))
            ark_index += 1
    if feats_dict:
        feats_path = os.path.join(os.path.abspath(feats_dir), 'fbank.%d.ark' % (ark_index))
        with open(feats_file, 'a') as f:
            f.write(feats_path + '\n')
            f.close()
        write_mat(feats_path, feats_dict)
        print("store %d utt fbank feats into %s" % (len(feats_dict), feats_path))
    print("finish handle %d wav files" % (num_wav))
    save_label_maps(label_map, map_path)
    print("finish save labels map")
