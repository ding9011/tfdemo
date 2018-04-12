#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
import random


def produce_labels_map(file_list):
    labels_dict = {}
    c_label = 0
    for item in file_list:
        label = item.strip().split(' ')[0].strip()
        if label in labels_dict.keys():
            pass
        else:
            labels_dict[label] = c_label
            c_label += 1
    return labels_dict

def shuffle_data(data_lists):
    dim = len(data_lists[0])
    for item in data_lists:
        if len(item) != dim:
            print("different data dim in lists")
            exit(1)
    index_shuf = range(len(data_lists[0]))
    random.shuffle(index_shuf)
    out_lists = []
    for _ in range(len(data_lists)):
        out_lists.append([])
    for i in range(len(out_lists)):
        for j in index_shuf:
            out_lists[i].append(data_lists[i][j])
    return out_lists

def neighbor_data(data, left_context, right_context, start_idx, end_idx):
    #data format: [num_frames, feature] for one utt
    #  print("param data shape:", data.shape, "left_context:", left_context, "right_context:", right_context, "start_idx:", start_idx, "end_idx:", end_idx)
    data_size = data.shape[0]
#    print("data size:", data_size)
    total_time = left_context + right_context + 1
    feature_dim = data.shape[1]
    split_datas = []
    i = start_idx
    while i < end_idx:
        if i < left_context:
            data_stack = np.tile(data[0, :], (left_context - i, 1))
            data_stack = np.row_stack((data_stack, data[0:i, :]))
        else:
            data_stack = data[i - left_context:i, :]
        data_stack = np.row_stack((data_stack, data[i:i + right_context + 1, :]))
        if i + right_context + 1 > data_size:
            data_stack = np.row_stack((data_stack, np.tile(data[-1, :], (i + right_context + 1 - data_size, 1))))
#            print("i + right_context + 1 - data_size:", i + right_context + 1 - data_size)
        if data_stack.shape != (total_time, feature_dim):
            print("data stack shape:", data_stack.shape)
            print(data_stack[-1, :])
            print(data[i + right_context, :])
            print("error when make data, data_stack shape:", data_stack.shape, "total_time:", total_time, "feature_dim:", feature_dim)
            exit(1)
        split_datas.append(data_stack)
        i += 1
    ret_data = np.array(split_datas)
    return ret_data

def produce_utt_samples(feat, window_size, window_shift, left_context, right_context):
    #feat shape: [num_frames, feature_dim]
    sample_index = 0
    ret_samples = []
    temp_feat = feat
    while temp_feat.shape[0] < window_size:
        temp_feat = np.row_stack((temp_feat, feat))
    while sample_index * window_shift + window_size < temp_feat.shape[0]:
        start_idx = sample_index * window_shift
        end_idx = sample_index * window_shift + window_size
        one_sample = neighbor_data(temp_feat, left_context, right_context, start_idx, end_idx)
        ret_samples.append(one_sample)
        sample_index += 1
    return ret_samples


def save_label_maps(label_maps, map_path):
    if not isinstance(label_maps, dict):
        print("error label maps input")
        exit(1)
    with open(map_path, 'w') as f:
        for k, v in label_maps.iteritems():
            f.write(str(k) + ' ' + str(v) + '\n')
        f.close()
    print("finish store label maps in " + map_path)

def read_label_maps(label_maps_file):
    if os.path.exists(label_maps_file):
        label_map = {}
        lines = open(label_maps_file).readlines()
        for line in lines:
            label = line.strip().split(' ')[0].strip()
            label_id = line.strip().split(' ')[1].strip()
            if not label in label_map.keys():
                label_map[label] = int(label_id)
        return label_map
    else:
        print("can't find the label map file:" + label_maps_file)
        exit(1)
