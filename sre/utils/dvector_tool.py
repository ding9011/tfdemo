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

def dvector_normalize_length(dvector):
    if len(dvector.shape) != 1:
        print("error shape of dvector:", dvector.shape)
        exit(1)
    l2_norm = np.sqrt(np.sum(np.square(dvector)))
    scale = l2_norm / np.sqrt(dvector.shape[0])
    #  print("scale:", scale)
    ret_dvector = np.divide(dvector, scale)
    return ret_dvector

def cosine_similarity(X, Y):
    if X.shape != Y.shape:
        print("error shape between X and Y, X shape:", X.shape, ", Y shape:", Y.shape)
        exit(1)
    division = np.sum(np.square(X)) * np.sum(np.square(Y))
    #  x = np.sqrt(np.sum(np.square(X)))
    #  y = np.sqrt(np.sum(np.square(Y)))
    ret = np.dot(X,Y) / np.sqrt(division)
    return ret

def compute_score1(X, Y, mat):
    if X.shape != Y.shape or X.shape[0] != mat.shape[0]:
        print("error shape between X and Y, X shape:", X.shape, ", Y shape:", Y.shape, ", mat shape:", mat)
        exit(1)
    X_ = np.dot(mat, X)
    Y_ = np.dot(mat, Y)
    score = np.dot(X_, Y_) / np.sqrt(np.dot(X_, X_) * np.dot(Y_, Y_))
    return score

def compute_eer(trials, enroll_dvectors, test_dvectors, transform_mat=None):
    print("enroll length: %d, test length: %d" % (len(enroll_dvectors), len(test_dvectors)))
    target_scores = []
    nontarget_scores = []
    if transform_mat is not None:
        print("using transform matrix")
    with open(trials, 'r') as f:
        for line in f:
            info = line.strip().split(' ')
            enroll_k = info[0]
            test_k = info[1]
            target_symbol = info[2]
            score = 0.0
            enroll_v = enroll_dvectors[enroll_k]
            test_v = test_dvectors[test_k]
            if transform_mat is None:
                score = cosine_similarity(enroll_v, test_v)
            else:
                score = compute_score1(enroll_v, test_v, transform_mat)
            if target_symbol == "target":
                target_scores.append(score)
            else:
                nontarget_scores.append(score)
    target_scores.sort()
    nontarget_scores.sort(reverse=True)

    #  print("target_scores:",target_scores)
    #  print("nontarget_scores:",nontarget_scores)
    #  print("target_trials:", target_trials)
    #  print("nontarget_trials:", nontarget_trials)
    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)
    eer = 0.0
    eer_th = 0.0
    num_err = 0
    for i in range(0, target_size):
        FR = i
        FA = 0
        threshold = target_scores[i]
        for score in nontarget_scores:
            if float(score) < float(threshold):
                break
            else:
                FA += 1
        FA_R = float(FA) / float(nontarget_size)
        FR_R = float(FR) / float(target_size)
        #  print("FAR: %.6f, FRR:%.6f, threshold=%.4f" % (FA_R, FR_R, threshold))
        if FA_R <= FR_R:
            eer = FR_R
            eer_th = threshold
            num_err = FA + FR
            break
        if FA_R <= 0:
            num_err = FA + FR
            break
    print("trials size:", target_size + nontarget_size)
    print("total error:", num_err)
    print("FA:", FA)
    return eer, eer_th

