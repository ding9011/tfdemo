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
from utils.log import get_logger
import os
import time
from functools import reduce
from operator import mul

np.set_printoptions(threshold='nan')
from utils.config import *
init_config("config/my.conf")



if not set_section("train"):
    print("can't find this configuration section")
    exit(1)
tf.app.flags.DEFINE_bool('use_gpu', bool(get("use_gpu")), 'use gpu?')
tf.app.flags.DEFINE_bool('appoint_gpu', bool(get("appoint_gpu")), 'appoint gpu?')
tf.app.flags.DEFINE_integer('num_gpu', int(get("num_gpu")), 'the number of gpu')



if not set_section("train"):
    print("can't find this configuration section")
    exit(1)

tf.app.flags.DEFINE_string('train_dir', str(get("train_dir")), 'train model store path')
tf.app.flags.DEFINE_string('train_data_file', str(get("train_data_file")), 'train data file path')
tf.app.flags.DEFINE_string('dev_data_file', str(get("dev_data_file")), 'dev data file path')
tf.app.flags.DEFINE_string('label_map_file', str(get("label_map_file")), 'speaker id map file path')
tf.app.flags.DEFINE_string('num_frames_file', str(get("num_frames_file")), 'the number of frames each utt')
tf.app.flags.DEFINE_string('train_log_dir', str(get("train_log_dir")), 'store training log')
tf.app.flags.DEFINE_string('dev_accuracy_log', str(get("dev_accuracy_log")), 'dev accuracy log path')
tf.app.flags.DEFINE_bool('dev_trials_symbol', bool(get("dev_trials_symbol")), 'whether generate trials')
tf.app.flags.DEFINE_string('dev_trials_name', str(get("dev_trials_name")), 'dev data trials name')
tf.app.flags.DEFINE_integer('num_dev_utt', int(get("num_dev_utt")), 'the number of dev utt')
tf.app.flags.DEFINE_integer('num_preload_samples', int(get("num_preload_samples")), "the number of preloading samples")


tf.app.flags.DEFINE_float('learning_rate', float(get("start_learning_rate")),
                          'Initial learning rate')

tf.app.flags.DEFINE_float('end_learning_rate', float(get("end_learning_rate")),
                          'The minimal end learning rate')

tf.app.flags.DEFINE_string('learning_rate_decay_type', str(get("learning_rate_decay_type")),
                           'Specifies how the learning rate is decayed. One of "fixed", "exponential", "polynomial"')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', float(get("learning_rate_decay_factor")),
                           'Learning decay factor')

tf.app.flags.DEFINE_string('optimizer', str(get("optimizer")), 'Specifies the optimizer format')
tf.app.flags.DEFINE_float('momentum', float(get("momentum")), 'Specifies the momentum param')

tf.app.flags.DEFINE_float('opt_epsilon', float(get("opt_epsilon")), 'a current good choice is 1.0 or 0.1 in ImageNet example')

tf.app.flags.DEFINE_integer('batch_size', int(get("batch_size")),
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('num_small_test_batch_size', int(get("num_small_test_batch_size")),
                            'number of batch in small dataset test')

tf.app.flags.DEFINE_bool('small_dataset_test', bool(get("small_dataset_test")),
                         'whether doing small dataset test.')

tf.app.flags.DEFINE_float('num_epochs_per_decay', float(get("num_epochs_per_decay")),
                          'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer('num_epochs', int(get("num_epochs")),
                            'The number of epochs for training')

tf.app.flags.DEFINE_integer('lstm_hidden_units', int(get("lstm_hidden_units")), 'number of lstm cell hidden untis')
tf.app.flags.DEFINE_integer('lstm_num_layers', int(get("lstm_num_layers")), 'number of lstm layers')
tf.app.flags.DEFINE_integer('feature_dim', int(get("feature_dim")), 'dim of feature')
tf.app.flags.DEFINE_integer('left_context', int(get("left_context")), 'number of left context')
tf.app.flags.DEFINE_integer('right_context', int(get("right_context")), 'number of right context')
tf.app.flags.DEFINE_integer('cnn_num_filter', int(get("cnn_num_filter")), 'define number of cnn filter, lstm_time must be divided exactly of this number, using in conv2d')
tf.app.flags.DEFINE_integer('cnn_shift_time', int(get("cnn_shift_time")), 'cnn depth stride time, using in conv3d')
tf.app.flags.DEFINE_integer('affine_feature_dim', int(get("affine_feature_dim")), 'residual feature dim')
tf.app.flags.DEFINE_integer('dvector_dim', int(get("dvector_dim")), 'dvector dim')
tf.app.flags.DEFINE_float('dropout', float(get("dropout_rate")), 'parameter of dropout')
tf.app.flags.DEFINE_bool('training', True, 'model state')
tf.app.flags.DEFINE_bool('batch_norm', bool(get("batch_norm")), 'doing batch normalization')

tf.app.flags.DEFINE_integer('num_step_store_model', int(get("num_step_store_model")), 'store model per num_step_store steps')

if not set_section("sample"):
    print("can't find this configuration section")
    exit(1)

tf.app.flags.DEFINE_integer('lstm_time', int(get("window_size")), 'lstm max time')
tf.app.flags.DEFINE_integer('window_size', int(get("window_size")), 'the number of frames in one sample')
tf.app.flags.DEFINE_integer('window_shift', int(get("window_shift")), 'the number of frames in utt frames shift')

FLAGS = tf.app.flags.FLAGS



def _configure_learning_rate(num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size * FLAGS.num_epochs_per_decay)

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized.', FLAGS.learning_rate_decay_type)

def _configure_optimizer(learning_rate):
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate,
                    rho=FLAGS.adadelta_rho,
                    epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
                    learning_rate,
                    initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
                    learning_rate,
                    epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
                    learning_rate,
                    learning_rate_power=FLAGS.ftrl_learning_rate_power,
                    initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
                    l1_regularization_strength=FLAGS.ftrl_l1,
                    l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    momentum=FLAGS.momentum,
                    name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
                    learning_rate,
                    decay=FLAGS.rmsprop_decay,
                    momentum=FLAGS.rmsprop_momentum,
                    epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer

def __generate_trials(trials, enroll_dict, test_dict):
    with open(trials, 'w') as f:
        for k1,v1 in enroll_dict.iteritems():
            spk1 = k1.strip().split('_')[0]
            for k2,v2 in test_dict.iteritems():
                spk2 = k2.strip().split('_')[0]
                if spk1 == spk2:
                    f.write(k1.strip() + ' ' + k2.strip() + ' target\n')
                else:
                    f.write(k1.strip() + ' ' + k2.strip() + ' nontarget\n')


def _generate_trials(trials, enroll_dict, test_dict=None):
    if test_dict == None:
        __generate_trials(trials, enroll_dict, enroll_dict)
    else:
        __generate_trials(trials, enroll_dict, test_dict)


def _valid_model(_sess, _embedding1, _embedding2, _inputs, _labels, _dev_accuracy_log, _dev_utt, _dev_trials_symbol, _epoch, _step, _FLAGS):
    spk_dvectors = {}
    batch_size = int(_FLAGS.batch_size)
    handled_utt = 0
    dev_batchs = []
    dev_labels = []
    dev_keys = []
    for k,v in _dev_utt.iteritems():
        dev_samples = produce_utt_samples(v, _FLAGS.window_size, _FLAGS.window_shift, _FLAGS.left_context, _FLAGS.right_context)
        for sample in dev_samples:
            dev_batchs.append(sample)
            dev_labels.append(1)
            dev_keys.append(k)
    num_dev_samples = len(dev_labels)
    print('the number of dev samples:', num_dev_samples)
    dev_dvectors = []
    for i in range(int(num_dev_samples / batch_size)):
        batch_datas = dev_batchs[i * batch_size: (i + 1) * batch_size]
        batch_labels = dev_labels[i * batch_size: (i + 1) * batch_size]
        em1s, em2s = _sess.run([_embedding1, _embedding2], feed_dict={_inputs: batch_datas, _labels: batch_labels})
        for j in range(batch_size):
            em1 = em1s[j]
            em2 = em2s[j]
            em1 = dvector_normalize_length(np.reshape(em1, [_FLAGS.dvector_dim]))
            em2 = dvector_normalize_length(np.reshape(em2, [_FLAGS.dvector_dim]))
            spk_dvector = np.divide(np.add(em1, em2), 2.0)
            spk_dvector = dvector_normalize_length(spk_dvector)
            dev_dvectors.append(spk_dvector)
    spk_temp_dvectors = {}
    for i in range(len(dev_dvectors)):
        if dev_keys[i] in spk_temp_dvectors.keys():
            spk_temp_dvectors[dev_keys[i]].append(dev_dvectors[i])
        else:
            spk_temp_dvectors[dev_keys[i]] = []
            spk_temp_dvectors[dev_keys[i]].append(dev_dvectors[i])
    for k,v in spk_temp_dvectors.iteritems():
        spk_dvector = np.mean(v, axis=0)
        if spk_dvector.shape[0] == 0:
            print("error speaker " + k + " dvector:", speaker_dvector)
            exit(1)
        if k in spk_dvectors.keys():
            print("duplicate keys:", k)
            exit(1)
        spk_dvectors[k] = spk_dvector
        print("finished computing " + k + " dvector")
    print("the number of dev data set utt:", len(spk_dvectors.keys()))
    dev_trials = _FLAGS.train_log_dir.rstrip('/') + '/' + _FLAGS.dev_trials_name
    if _dev_trials_symbol:
        _generate_trials(dev_trials, spk_dvectors)
    eer, eer_th = compute_eer(dev_trials, spk_dvectors, spk_dvectors)
    with open(_dev_accuracy_log, 'a') as f:
        f.write('epoch %d, global step:%d, eer: %.4f, threshold: %.4f\n' % (_epoch + 1, _step, eer, eer_th))
        f.close()
    print("eer: %.2f, threshold: %.2f" % (eer, eer_th))

def _get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        name = variable.name
        print(name, ' shape:', shape)
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


model_path = FLAGS.train_dir
if not os.path.exists(model_path):
    os.makedirs(model_path)
else:
    empty_dir(os.path.dirname(model_path))

if not os.path.exists(FLAGS.train_log_dir):
    os.makedirs(FLAGS.train_log_dir)

dev_accuracy_log = FLAGS.train_log_dir.rstrip('/') + '/' + FLAGS.dev_accuracy_log
if os.path.exists(dev_accuracy_log):
    with open(dev_accuracy_log, 'w') as f:
        f.write('')
        f.close()

log_path = os.path.join(os.path.abspath(FLAGS.train_log_dir), 'training_log')
if os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write('')
        f.close()
logger = get_logger(log_path)

label_map = read_label_maps(FLAGS.label_map_file)
num_speakers = len(label_map.keys())
print('the number of speakers:', num_speakers)
logger.info('the number of speakers: %d' % (num_speakers))

train_feats_file_list = open(FLAGS.train_data_file).readlines()
dev_feats_file_list = open(FLAGS.dev_data_file).readlines()

num_speakers = len(label_map.keys())
print("number of speakers:", num_speakers)
total_samples = 0
window_size = int(FLAGS.window_size)
window_shift = int(FLAGS.window_shift)
with open(FLAGS.num_frames_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        frames = int(line.strip().split(' ')[1])
        temp_frames = frames
        while temp_frames < window_size:
            temp_frames += frames
        total_samples += int(int(temp_frames - window_size) / window_shift) + 1
    f.close()
print("number of samples:", total_samples)
logger.info("number of samples:" + str(total_samples))
num_step_store_model = int(FLAGS.num_step_store_model)
#  time.sleep(100)

# load dev data
num_dev_utt = FLAGS.num_dev_utt
num_load_utt = 0
dev_utt = {}
for dev_feats_ark in dev_feats_file_list:
    file_path = dev_feats_ark.strip()
    dev_dcit = read_mat(file_path)
    for k,v in dev_dcit.iteritems():
        dev_utt[k] = v
        num_load_utt += 1
        if num_load_utt >= num_dev_utt:
            break
    if num_load_utt >= num_dev_utt:
        break

def main(_):
    #  tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        num_samples_per_epoch = total_samples
        if FLAGS.small_dataset_test:
            num_samples_per_epoch = FLAGS.batch_size * FLAGS.num_small_test_batch_size
        num_batches_per_epoch = int(num_samples_per_epoch / FLAGS.batch_size)
        print("number of batches:", num_batches_per_epoch)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = _configure_learning_rate(num_samples_per_epoch, global_step)
        #      boundaries = [2000, 5000, 8000, 12000, 15000, 18000, 20000, 25000]
        #      values = [0.001, 0.0005, 0.0003, 0.0001, 0.00005, 0.00003, 0.00001, 0.000005, 0.000003]
        #  learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        neighbor_dim = FLAGS.left_context + FLAGS.right_context + 1
        lstm_time = FLAGS.lstm_time
        device_type = 'cpu'
        if FLAGS.use_gpu:
            device_type = 'gpu'
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/' + device_type + ':0'):
                inputs = tf.placeholder(tf.float32, [None, lstm_time, neighbor_dim, FLAGS.feature_dim])
                labels = tf.placeholder(tf.int32, [None])
                #  logits, embeddings1, embeddings2 = prepare_model_res_GRU(inputs, num_speakers, FLAGS)
                #  logits, embeddings1, embeddings2 = prepare_model_BiGRU(inputs, num_speakers, FLAGS)
                logits, embeddings1, embeddings2 = prepare_model(inputs, num_speakers, FLAGS)
                softmax_result = tf.nn.softmax(logits)
                label_onehot = tf.one_hot(labels - 1, depth=num_speakers)
                opt = _configure_optimizer(learning_rate)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_onehot, name='loss')
                with tf.name_scope('loss'):
                    loss = tf.reduce_mean(cross_entropy)

                with tf.name_scope('result_print'):
                    judge = tf.argmax(logits, 1)
                    true_judge = tf.argmax(label_onehot, 1)
                    prob = tf.nn.softmax(logits)

                with tf.name_scope('train_op'):
                    train_op = opt.minimize(loss, global_step=global_step, aggregation_method=2)
                with tf.name_scope('accuracy'):
                    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label_onehot, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        summaries.add(tf.summary.scalar('global_step', global_step))
        summaries.add(tf.summary.scalar('eval/Loss', loss))
        summaries.add(tf.summary.scalar('accuracy', accuracy))
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        summary_merged = tf.summary.merge_all()

    #  check_point_dir = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'train_logs-1200')
    #  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        saver = tf.train.Saver(max_to_keep=100)
        summary_writer = tf.summary.FileWriter(model_path, graph=graph)
        sess.run(tf.global_variables_initializer())
        #  sess.run(tf.local_variables_initializer())
        #  saver.restore(sess, check_point_dir)
        num_params = _get_num_params()
        print("the number of parameters:", num_params)
        logger.info("the number of parameters:" + str(num_params))
        print("watching model size")
        time.sleep(10)

        preload_batch_datas = []
        preload_batch_labels = []
        # small data for test
        dev_trials_symbol = FLAGS.dev_trials_symbol
        if FLAGS.small_dataset_test:
            print("It's in small dataset test")
            feats_index = 0
            while len(preload_batch_labels) < int(FLAGS.num_small_test_batch_size * FLAGS.batch_size):
                feats_file = train_feats_file_list[feats_index].strip()
                feats = read_mat(feats_file)
                for k,v in feats.iteritems():
                    label = k.strip().split('_')[0]
                    label_id = label_map[label]
                    samples = produce_utt_samples(v, window_size, window_shift, FLAGS.left_context, FLAGS.right_context)
                    for i in range(len(samples)):
                        preload_batch_datas.append(samples[i])
                        preload_batch_labels.append(label_id)
                        if len(preload_batch_labels) >= int(FLAGS.num_small_test_batch_size * FLAGS.batch_size):
                            break
                    if len(preload_batch_labels) >= int(FLAGS.num_small_test_batch_size * FLAGS.batch_size):
                        break
                feats_index += 1
                if feats_index >= len(train_feats_file_list):
                    print("preload samples is too big")
                    exit(1)
            print("the number of preload samples:", len(preload_batch_labels))

            for epoch in range(FLAGS.num_epochs):
                shuffle_datas = shuffle_data([preload_batch_datas, preload_batch_labels])
                preload_batch_datas = shuffle_datas[0]
                preload_batch_labels = shuffle_datas[1]
                for batch_num in range(FLAGS.num_small_test_batch_size):
                    start_idx = batch_num * FLAGS.batch_size
                    end_idx = (batch_num + 1) * FLAGS.batch_size
                    batch_datas = np.array(preload_batch_datas[start_idx: end_idx])
                    batch_labels = preload_batch_labels[start_idx: end_idx]
                    _, loss_value, train_accuracy, summary, out_judge, \
                    out_true_judge, out_prob, out_learning_rate, softmax_out, step = sess.run(
                        [train_op, loss, accuracy, summary_merged, \
                        judge, true_judge, prob, learning_rate, softmax_result, global_step], \
                        feed_dict={inputs: batch_datas, labels: batch_labels})
                    summary_writer.add_summary(summary, step)
                    print("Epoch " + str(epoch + 1) + ", Minibatch " + str(batch_num + 1) + \
                        " of %d " % num_batches_per_epoch + ", Minibatch Loss=" + "{:.4f}".format(loss_value) + \
                        ", TRAIN ACCURACY=" + "{:.3f}".format(100 * train_accuracy))
                    print("global step:", step)
                    #  print("softmax result:", softmax_out[:])
                    print("mean softmax result:", np.mean(np.max(softmax_out, axis=1)))
                    print("program out labels:", out_judge + 1)
                    print("true lables:", out_true_judge + 1)
                    print("learning rate:", out_learning_rate)
            exit(1)

        last_loss = -1.0
        num_preload_samples = FLAGS.num_preload_samples
        step = 0
        train_feats_files = shuffle_data([train_feats_file_list])[0]
        for epoch in range(FLAGS.num_epochs):
            train_feats_files = shuffle_data([train_feats_files])[0]
            print("train_feats_files lines:", len(train_feats_files))
            feats_file_index = 0
            feats_key_index = 0
            feats_dict = {}
            feats_keys = []
            for batch_num in range(num_batches_per_epoch):
                if len(preload_batch_labels) < FLAGS.num_preload_samples / 2 and feats_file_index < len(train_feats_files):
                    while len(preload_batch_labels) < FLAGS.num_preload_samples:
                        if feats_key_index >= len(feats_keys) or feats_key_index == 0:
                            feats_dict = read_mat(train_feats_files[feats_file_index].strip())
                            feats_keys = feats_dict.keys()
                            feats_file_index += 1
                            feats_key_index = 0
                        spk = feats_keys[feats_key_index].strip().split('_')[0]
                        label_id = label_map[spk]
                        samples = produce_utt_samples(feats_dict[feats_keys[feats_key_index]], window_size, window_shift, FLAGS.left_context, FLAGS.right_context)
                        for i in range(len(samples)):
                            preload_batch_datas.append(samples[i])
                            preload_batch_labels.append(label_id)
                        feats_key_index += 1
                        if feats_file_index >= len(train_feats_files) and feats_key_index >= len(feats_keys):
                            shuffle_datas = shuffle_data([preload_batch_datas, preload_batch_labels])
                            preload_batch_datas = shuffle_datas[0]
                            preload_batch_labels = shuffle_datas[1]
                            break
                    shuffle_datas = shuffle_data([preload_batch_datas, preload_batch_labels])
                    preload_batch_datas = shuffle_datas[0]
                    preload_batch_labels = shuffle_datas[1]
                if len(preload_batch_labels) < FLAGS.batch_size:
                    break
                print("the number of preload samples:", len(preload_batch_labels))
                print("feat files index: %d, feat keys index:%d" % (feats_file_index, feats_key_index))

                batch_datas, batch_labels = preload_batch_datas[0:FLAGS.batch_size], preload_batch_labels[0:FLAGS.batch_size]
                print("the number of batch_datas:", len(batch_datas))
                _, loss_value, train_accuracy, summary, out_judge, \
                out_true_judge, out_prob, out_learning_rate, softmax_out, step = sess.run(
                        [train_op, loss, accuracy, summary_merged, \
                        judge, true_judge, prob, learning_rate, softmax_result, global_step], \
                        feed_dict={inputs: batch_datas, labels: batch_labels})

                preload_batch_datas = preload_batch_datas[FLAGS.batch_size:]
                preload_batch_labels = preload_batch_labels[FLAGS.batch_size:]

            #      print("Epoch " + str(epoch + 1) + ", Minibatch " + str(batch_num + 1) + \
            #              " of %d " % num_batches_per_epoch)
            #  exit(1)

                print("Epoch " + str(epoch + 1) + ", Minibatch " + str(batch_num + 1) + \
                        " of %d " % num_batches_per_epoch + ", Minibatch Loss=" + "{:.4f}".format(loss_value) + \
                        ", TRAIN ACCURACY=" + "{:.3f}".format(100 * train_accuracy))
                #  print(batch_datas)
                #  print(batch_labels)
                print("global step:", step)
                #  print("softmax result:", softmax_out[:])
                print("mean softmax result:", np.mean(np.max(softmax_out, axis=1)))
                print("program out labels:", out_judge + 1)
                print("true lables:", out_true_judge + 1)
                print("learning rate:", out_learning_rate)
                summary_writer.add_summary(summary, step)
                if num_step_store_model > 0 and step > 0 and step % num_step_store_model == 0:
                    saver.save(sess, model_path, global_step=step)
                    _valid_model(sess, embeddings1, embeddings2, inputs, labels, dev_accuracy_log, dev_utt, dev_trials_symbol, epoch, step, FLAGS)
                    if dev_trials_symbol:
                        dev_trials_symbol = False

            #  save and test each epoch
            if num_step_store_model <= 0:
                saver.save(sess, model_path, global_step=step)
                _valid_model(sess, embeddings1, embeddings2, inputs, labels, dev_accuracy_log, dev_utt, dev_trials_symbol, epoch, step, FLAGS)
                if dev_trials_symbol:
                    dev_trials_symbol = False

if __name__ == "__main__":
    tf.app.run()




