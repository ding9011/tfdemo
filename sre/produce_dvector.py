#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tables as tb
import tensorflow as tf
import numpy as np
from utils.data_tool import *
from utils.dvector_tool import *
from utils.dvector_io import *
from model.model import *
from utils.tools import *
import os
import utils.dvector_io as dvector_io
import sys
from datetime import datetime
import time

np.set_printoptions(threshold='nan')
from utils.config import *
init_config("config/my.conf")

if not set_section("produce"):
    print("can't find produce configuration section")
    exit(1)
tf.app.flags.DEFINE_string('train_dir', str(get("train_dir")), 'train model store path')
tf.app.flags.DEFINE_string('check_point', str(get("check_point")), 'train model store step')
tf.app.flags.DEFINE_string('num_speakers', str(get("num_speakers")), 'equal to number of speaker in train dataset, which is used to restore model')

tf.app.flags.DEFINE_string('test_file', str(get("test_file")), 'produce dvector data set file path')
tf.app.flags.DEFINE_bool('kaldi_prepare_symbol', bool(get("kaldi_prepare_symbol")), "generate trials file")
tf.app.flags.DEFINE_string('kaldi_prepare_path', str(get("kaldi_prepare_path")), 'kaldi data prepare directory path')
tf.app.flags.DEFINE_string('utt2spk_suffix', str(get("utt2spk_suffix")), 'store dvector file path with kaldi_prepare_path')
tf.app.flags.DEFINE_string('target_file', str(get("target_file")), 'store dvector file path with kaldi_prepare_path')

tf.app.flags.DEFINE_integer('batch_size', 1,
                            'The number of samples in each batch. To simulate shuffling input data ')

if not set_section("train"):
    print("can't find train configuration section")
    exit(1)
tf.app.flags.DEFINE_integer('lstm_hidden_units', int(get("lstm_hidden_units")), 'number of lstm cell hidden untis')
tf.app.flags.DEFINE_integer('lstm_num_layers', int(get("lstm_num_layers")), 'number of lstm layers')
tf.app.flags.DEFINE_integer('feature_dim', int(get("feature_dim")), 'dim of feature')
tf.app.flags.DEFINE_integer('left_context', int(get("left_context")), 'number of left context')
tf.app.flags.DEFINE_integer('right_context', int(get("right_context")), 'number of right context')
tf.app.flags.DEFINE_integer('cnn_num_filter', int(get("cnn_num_filter")), 'define number of cnn filter, lstm_time must be divided exactly of this number, using in conv2d')
tf.app.flags.DEFINE_integer('cnn_shift_time', int(get("cnn_shift_time")), 'cnn depth stride time, using in conv3d')
tf.app.flags.DEFINE_integer('affine_feature_dim', int(get("affine_feature_dim")), 'residual feature dim')
tf.app.flags.DEFINE_integer('dvector_dim', int(get("dvector_dim")), 'dvector dim')
tf.app.flags.DEFINE_bool('dropout_symbol', bool(get("dropout_symbol")), 'whether applying dropout layer')
tf.app.flags.DEFINE_float('dropout', float(get("dropout_rate")), 'probability to keep units in cnn')
tf.app.flags.DEFINE_bool('training', False, 'model state')
tf.app.flags.DEFINE_bool('batch_norm', bool(get("batch_norm")), 'doing batch normalization')

if not set_section("sample"):
    print("cant find sample configuration section")
    exit(1)
tf.app.flags.DEFINE_integer('lstm_time', int(get("window_size")), 'lstm max time')
tf.app.flags.DEFINE_integer('window_size', int(get("window_size")), 'the number of frames in one sample')
tf.app.flags.DEFINE_integer('window_shift', int(get("window_shift")), 'the number of frames in utt frames shift')

FLAGS = tf.app.flags.FLAGS


model_path = FLAGS.train_dir



test_feats_file_list = open(FLAGS.test_file, 'r').readlines()

num_speakers = int(FLAGS.num_speakers)

#model file path
check_point_dir = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'train_logs-' + FLAGS.check_point)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()
    with graph.as_default(), tf.device('/gpu:0'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        neighbor_dim = FLAGS.left_context + FLAGS.right_context + 1
        lstm_time = FLAGS.lstm_time
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, lstm_time, neighbor_dim, FLAGS.feature_dim])
                labels = tf.placeholder(tf.int32, [FLAGS.batch_size])
                #  _, embeddings1, embeddings2 = prepare_model_BiGRU(inputs, num_speakers, FLAGS)
                _, embeddings1, embeddings2 = prepare_model_res_GRU(inputs, num_speakers, FLAGS)


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
#        sess.run(tf.local_variables_initializer())
        saver.restore(sess, check_point_dir)
        step = 0

        test_dvectors = {}
        num_test_utt = 0
        for test_feats_ark in test_feats_file_list:
            ark_path = test_feats_ark.strip()
            test_feats_dicts = read_mat(ark_path)
            for k,v in test_feats_dicts.iteritems():
                start_time = time.time()
                samples = produce_utt_samples(v, FLAGS.window_size, FLAGS.window_shift, FLAGS.left_context, FLAGS.right_context)
                spk_dvectors = []
                for sample in samples:
                    em1, em2 = sess.run([embeddings1, embeddings2], feed_dict={inputs:[sample], labels:[1]})
                    em1 = dvector_normalize_length(np.reshape(em1, [FLAGS.dvector_dim]))
                    em2 = dvector_normalize_length(np.reshape(em2, [FLAGS.dvector_dim]))
                    spk_dvector = np.divide(np.add(em1, em2), 2.0)
                    spk_dvector = dvector_normalize_length(spk_dvector)
                    spk_dvector = dvector_normalize_length(np.reshape(em1, [FLAGS.dvector_dim]))
                    if spk_dvector.shape[0] != FLAGS.dvector_dim:
                        continue
                    spk_dvectors.append(spk_dvector)
                if len(spk_dvectors) == 0:
                    print("error dvector extract for utt:" + k)
                    continue
                dvector = np.mean(spk_dvectors, axis=0)
                if dvector.shape[0] != FLAGS.dvector_dim:
                    print("error dvector extract for utt:" + k)
                    continue
                test_dvectors[k] = dvector
                print("finish speaker [%s] utt dvector" % (k))
                end_time = time.time()
                print("used time: " + str((end_time - start_time)) + "s")
                num_test_utt += 1

        print("finish extract %d utt dvectors" % (num_test_utt))
        if not os.path.exists(FLAGS.kaldi_prepare_path):
            os.makedirs(FLAGS.kaldi_prepare_path)
        dvector_file = FLAGS.kaldi_prepare_path.rstrip('/') + '/' + FLAGS.target_file
        if not dvector_io.write_vec(dvector_file, test_dvectors):
            exit(1)
        print("finish write vec to " + FLAGS.target_file)
        if FLAGS.kaldi_prepare_symbol:
            if not dvector_io.write_to_kaldi_prepare(FLAGS.kaldi_prepare_path, test_dvectors, FLAGS.utt2spk_suffix):
                exit(1)
            print("finish write utt2spk to " + FLAGS.kaldi_prepare_path)
        #  eer, eer_th = compute_eer(test_dvectors, test_dvectors)
        #  print("eer: %.4f, threshold: %.4f" % (eer, eer_th))



if __name__ == "__main__":
    tf.app.run()




