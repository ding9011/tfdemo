#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tables as tb
import numpy as np
import os
import copy
import scipy
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datetime import datetime
import time
np.set_printoptions(threshold='nan')

def norm2(dvector):
    dim = dvector.shape[0]
    l2_norm = np.sqrt(np.sum(np.square(dvector)))
    scale = l2_norm / np.sqrt(dim)
    #  print("scale:", scale)
    ret_dvector = np.divide(dvector, scale)
    return ret_dvector

def compute_mean_dict(vec_dict):
    dim = vec_dict[vec_dict.keys()[0]].shape[0]
    num_vec = len(vec_dict)
    mean = np.zeros(dim)
    for key, vec in vec_dict.iteritems():
        mean += vec / num_vec
    return mean

def svm_kernel1(X, Y, mat):
    X_ = np.dot(mat, X)
    Y_ = np.dot(mat, Y)
    score = np.dot(X_, Y_) / np.sqrt(np.dot(X_, X_) * np.dot(Y_, Y_))
    return score

def within_class_conv_matrix(vec_dict):
    matrix_speaker = {}
    dim = 0
    for key, vec in vec_dict.iteritems():
        speaker_id = key.strip().split('_')[0]
        if dim == 0:
            dim = vec.shape[0]
        if speaker_id in matrix_speaker.keys():
            matrix_speaker[speaker_id].append(vec)
        else:
            matrix_speaker[speaker_id] = [vec]
    num_speaker = len(matrix_speaker)
    print("number of speakers:", num_speaker)
    W = []
    for key, vecs in matrix_speaker.iteritems():
        num_utt = len(vecs)
        matrix = np.array(vecs, dtype=np.float64)
        mean = np.mean(matrix, axis=0)
        w_speaker = []
        for vec in vecs:
            w_speaker.append(np.outer((vec - mean), (vec - mean)))
        W.append(np.mean(np.array(w_speaker, dtype=np.float64), axis=0))
    W_matrix = np.mean(np.array(W, dtype=np.float64), axis=0)
    return W_matrix


def matrix_cholesky(vec_dict):
    W_matrix = within_class_conv_matrix(vec_dict)
    B = np.linalg.cholesky(W_matrix)
    print("finish train matrix, matrix shape:", B.shape)
    return B

def matrix_eig(vec_dict, k):
    W_matrix = within_class_conv_matrix(vec_dict)
    w, v= np.linalg.eig(W_matrix)
    sorted_indices = np.argsort(w)
    R = v[:, sorted_indices[:-k-1:-1]]
    #  U, s, V = np.linalg.svd(W_matrix, full_matrices=False)
    print("topk_v shape:", R.shape)
    I = np.eye(R.shape[0])
    P = I - np.dot(R, R.T)
    P = np.real(P)
    #  R = np.dot(U, V)
    #  print("finish train matrix, matrix shape:", R.shape)
    print("P shape:", P.shape)
    return P

def train_lda(vec_dict):
    matrix_speaker = {}
    for key, vec in vec_dict.iteritems():
        speaker_id = key.strip().split('_')[0]
        if speaker_id in matrix_speaker.keys():
            matrix_speaker[speaker_id].append(vec)
        else:
            matrix_speaker[speaker_id] = [vec]
    num_speaker = len(matrix_speaker)
    print("number of speakers:", num_speaker)
    index = 1
    X = []
    Y = []
    for _, vecs in matrix_speaker.iteritems():
        for vec in vecs:
            X.append(vec)
            Y.append(index)
        index += 1
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, Y)
    print("finish train lda")
    return clf


def train_svm(cholesky_mat, train_dict):
    clf = svm.SVC(kernel=svm_kernel1)
    X = []
    Y = []
    Z = []
    start_time = time.time()
    for enroll_k, enroll_v in train_dict.iteritems():
        enroll_spk = enroll_k.strip().split('_')[0]
        for (test_k, test_v) in train_dict.iteritems():
            test_spk = test_k.strip().split('_')[0]
            X.append(enroll_v)
            Y.append(test_v)
            if test_spk == enroll_spk:
                Y.append(1)
            else:
                Y.append(0)
    end_time = time.time()
    print("make train data used time: " + str((end_time - start_time)) + "s")
    clf.fit(X, Y, Z)
    print("finish train svm")
    return clf
