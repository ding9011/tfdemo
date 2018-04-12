#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tables as tb
import tensorflow as tf
import numpy as np
import os


def PReLU(inputs, scope):
    alphas = tf.get_variable(scope, inputs.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    return tf.nn.relu(inputs) + alphas * (inputs - abs(inputs)) * 0.5

def weight_variable(shape, name='weights'):
    #  initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

def bias_variable(shape, name='biases'):
    #  initializer = tf.constant_initializer(0.1)
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

def batch_norm_prelu(FLAGS, inputs, scope):
    inputs = tf.contrib.layers.batch_norm(inputs, decay=0.9, center=True, scale=True, updates_collections=None, is_training=FLAGS.training, scope=scope)
    return PReLU(inputs, scope + '_prelu')

def batch_norm_relu(FLAGS, inputs, scope):
    inputs = tf.contrib.layers.batch_norm(inputs, decay=0.9, center=True, scale=True, updates_collections=None, is_training=FLAGS.training, scope=scope)
    return tf.nn.relu(inputs)

def batch_norm_tanh(FLAGS, inputs, scope):
    inputs = tf.contrib.layers.batch_norm(inputs, decay=0.9, center=True, scale=True, updates_collections=None, is_training=FLAGS.training, scope=scope)
    return tf.tanh(inputs)


def affine_layer_prelu(FLAGS, inputs, affine_scope, reuse_symbol, feature_dim, affine_dim):
    with tf.variable_scope(affine_scope, reuse=reuse_symbol) as scope:
        print(affine_scope + " inputs shape:", inputs.shape)
        w = weight_variable([feature_dim, affine_dim], 'affine_w')
        b = weight_variable([affine_dim], 'affine_b')
        fc = tf.add(tf.matmul(inputs, w), b)
        return PReLU(fc, 'affine_prelu')

def affine_layer_tanh(FLAGS, inputs, affine_scope, reuse_symbol, feature_dim, affine_dim):
    with tf.variable_scope(affine_scope, reuse=reuse_symbol) as scope:
        print(affine_scope + " inputs shape:", inputs.shape)
        w = weight_variable([feature_dim, affine_dim], 'affine_w')
        b = weight_variable([affine_dim], 'affine_b')
        fc = tf.add(tf.matmul(inputs, w), b)
        return tf.tanh(fc)

def affine_layer_relu(FLAGS, inputs, affine_scope, reuse_symbol, feature_dim, affine_dim):
    with tf.variable_scope(affine_scope, reuse=reuse_symbol) as scope:
        print(affine_scope + " inputs shape:", inputs.shape)
        w = weight_variable([feature_dim, affine_dim], 'affine_w')
        b = weight_variable([affine_dim], 'affine_b')
        fc = tf.add(tf.matmul(inputs, w), b)
        return tf.nn.relu(fc)

def my_conv2d(FLAGS, x, W, b, scope, strides = [1, 1, 1, 1], padding = "VALID"):
    x = tf.nn.conv2d(x, W, strides=strides, padding=padding, name=scope)
    x = tf.nn.bias_add(x, b)
    return x

def my_conv3d(FLAGS, x, W, b, scope, strides = [1, 1, 1, 1, 1], padding = "VALID"):
    x = tf.nn.conv3d(x, W, strides=strides, padding=padding, name=scope)
    x = tf.nn.bias_add(x, b)
    return x

def res_block(FLAGS, inputs, filters_out, res_scope):
    """
    # inputs = [batch_size * lstm_time, neighbor_dim, feature_dim, filters_in]
    """
    with tf.variable_scope(res_scope) as scope:
        filters_in = int(inputs.shape[-1])
        #  print(res_scope + ' filter_in:', filters_in)
        weights = {
            'w1': weight_variable([1, 3, 3, filters_in, filters_out], 'res_w1'),
            'w2': weight_variable([1, 3, 3, filters_out, filters_out], 'res_w2'),
        }
        wc1_hist = tf.summary.histogram('res_block_weights_1', weights['wc1'])
        wc2_hist = tf.summary.histogram('res_block_weights_2', weights['wc2'])

        biases = {
            'b1': bias_variable([filters_out], 'res_b1'),
            'b2': bias_variable([filters_out], 'res_b2'),
        }

        strides_in = [1, 1, 1, 1, 1]
        if filters_in != filters_out:
            strides_in = [1, 1, 2, 2, 1]

        orig_x = inputs
        x = batch_norm_relu(FLAGS, inputs, 'res_block_inputs')
        # sub layer 1
        x = my_conv3d(FLAGS, x, weights['w1'], biases['b1'], 'sub_layer_1_conv1', strides=strides_in, padding='SAME')
        x = batch_norm_relu(FLAGS, x, 'sub_layer_1_bn')
        x = my_conv3d(FLAGS, x, weights['w2'], biases['b2'], 'sub_layer_1_conv2', padding='SAME')
        x = batch_norm_relu(FLAGS, x, 'sub_layer_2_bn')

        with tf.variable_scope('shortcut'):
            if filters_in != filters_out:
                orig_x = tf.nn.max_pool3d(orig_x, strides_in, strides_in, padding='SAME')
                orig_x = tf.pad(
                        orig_x, [[0, 0], [0, 0], [0, 0], [(filters_out - filters_in) // 2, (filters_out - filters_in) // 2]])
            x += orig_x
        return x

def res_bottleneck_block(FLAGS, inputs, filters_out, res_scope):
    """
    # inputs = [batch_size * lstm_time, neighbor_dim, feature_dim, filters_in]
    """
    with tf.variable_scope(res_scope) as scope:
        filters_in = int(inputs.shape[-1])
        weights = {
            'w1': weight_variable([1, 1, 1, filters_in, filters_out / 4], 'res_bottleneck_block_w1'),
            'w2': weight_variable([1, 3, 3, filters_out / 4, filters_out / 4], 'res_bottleneck_block_w2'),
            'w3': weight_variable([1, 1, 1, filters_out / 4, filters_out], 'res_bottleneck_block_w3'),
            'shortcut': weight_variable([1, 1, 1, filters_in, filters_out], 'res_bottleneck_block_shortcut_w'),
        }
        wc1_hist = tf.summary.histogram('res_bottleneck_block_weights_1', weights['w1'])
        wc2_hist = tf.summary.histogram('res_bottleneck_block_weights_2', weights['w2'])
        wc3_hist = tf.summary.histogram('res_bottleneck_block_weights_3', weights['w3'])
        shortcut_hist = tf.summary.histogram('res_bottleneck_block_shortcut', weights['shortcut'])

        biases = {
            'b1': bias_variable([filters_out / 4], 'res_bottleneck_block_b1'),
            'b2': bias_variable([filters_out / 4], 'res_bottleneck_block_b2'),
            'b3': bias_variable([filters_out], 'res_bottleneck_block_b3'),
            'shortcut': bias_variable([filters_out], 'res_bottleneck_block_shortcut_b'),
        }

        strides_in = [1, 1, 1, 1, 1]
        if filters_in != filters_out:
            strides_in = [1, 1, 2, 2, 1]

        orig_x = inputs
        x = batch_norm_relu(FLAGS, inputs, 'res_bottleneck_block_inputs')
        # sub layer 1
        x = my_conv3d(FLAGS, x, weights['w1'], biases['b1'], 'sub_layer_1_conv1', strides=strides_in, padding='SAME')
        x = batch_norm_relu(FLAGS, x, 'sub_layer_1_bn')
        x = my_conv3d(FLAGS, x, weights['w2'], biases['b2'], 'sub_layer_1_conv2', padding='SAME')
        x = batch_norm_relu(FLAGS, x, 'sub_layer_2_bn')
        x = my_conv3d(FLAGS, x, weights['w3'], biases['b3'], 'sub_layer_1_conv3', padding='SAME')
        x = batch_norm_relu(FLAGS, x, 'sub_layer_3_bn')

        with tf.variable_scope('shortcut'):
            if filters_in != filters_out:
                orig_x = my_conv3d(FLAGS, orig_x, weights['shortcut'], biases['shortcut'], 'shortcut', strides=strides_in, padding='SAME')
            x += orig_x
        return x

def stack_res_block(FLAGS, inputs, filters_out, block, num_block, scope):
    with tf.variable_scope(scope) as scope:
        x = inputs
        for i in range(num_block):
            x = block(FLAGS, x, filters_out, 'block_%d' % (i))
            x = batch_norm_relu(FLAGS, x, 'block_%d_bn' % (i))
        return x

def get_resnet(FLAGS, inputs, cnn_scope, reuse_symbol):
    """
    # inputs = [batch_size, lstm_time, neighbor_dim, feature_dim, 1]
    """
    with tf.variable_scope(cnn_scope, reuse=reuse_symbol) as scope:
        x = batch_norm_relu(FLAGS, inputs, 'resnet_inputs')
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            x = tf.layers.dropout(x, rate=FLAGS.dropout, training=FLAGS.training)
        filters= [64, 128, 256, 512]
        w_first = weight_variable([1, 5, 5, 1, 32], 'w_first')
        w_first_hist = tf.summary.histogram('res_first_layer_weights', w_first)
        b_first = bias_variable([32], 'b_first')
        s_first = [1, 1, 2, 2, 1]
        x = my_conv3d(FLAGS, x, w_first, b_first, 'first_layer', strides=s_first, padding='SAME')
        first_hist = tf.summary.histogram('res_first_layer', x)
        print('resnet first layer shape:', x.shape)
        x = stack_res_block(FLAGS, x, filters[0], res_bottleneck_block, 2, 'res_1')
        res1_hist = tf.summary.histogram('res_1_output', x)
        print('resnet blocks 1 shape:', x.shape)
        x = stack_res_block(FLAGS, x, filters[1], res_bottleneck_block, 2, 'res_2')
        res2_hist = tf.summary.histogram('res_2_output', x)
        print('resnet blocks 2 shape:', x.shape)
        x = stack_res_block(FLAGS, x, filters[2], res_bottleneck_block, 2, 'res_3')
        res3_hist = tf.summary.histogram('res_3_output', x)
        print('resnet blocks 3 shape:', x.shape)
        x = stack_res_block(FLAGS, x, filters[3], res_bottleneck_block, 3, 'res_4')
        res4_hist = tf.summary.histogram('res_4_output', x)
        print('resnet blocks 4 shape:', x.shape)
        #  x = tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name='resnet_fc_pool')
        #  print('resnet pool shape:', x.shape)
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            x = tf.layers.dropout(x, rate=FLAGS.dropout, training=FLAGS.training)
        w_fc = weight_variable([1, 1, 2, 512, 512], 'w_fc')
        fc_hist = tf.summary.histogram('res_w_fc_output', w_fc)
        b_fc = bias_variable([512], 'b_fc')
        x = my_conv3d(FLAGS, x, w_fc, b_fc, 'conv_fc_layer', padding='VALID')
        print('resnet conv fc shape:', x.shape)
        x = tf.reshape(x, [-1, 512])
        return x

def single_GRU(FLAGS, inputs, GRU_scope, reuse_symbol):
    """
        inputs shape = [batch_size, lstm_time, cnn_out]
        max_time = left_context + 1(current_frame) + right_context
        define lstm
    """
    with tf.variable_scope(GRU_scope, reuse=reuse_symbol) as scope:
        if not reuse_symbol:
            inputs_hist = tf.summary.histogram('inputs', inputs)
        print("GRU_inputs shape:", inputs.shape)
        x = batch_norm_relu(FLAGS, inputs, 'GRU_inputs')
        hidden_layer = tf.contrib.rnn.GRUCell(FLAGS.lstm_hidden_units)
        outputs, _ = tf.nn.dynamic_rnn(hidden_layer, inputs, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1,0,2])
        last = outputs[-1]
        last = tf.nn.relu(last)
        print('GRU outputs shape:', last.shape)
        fc = affine_layer_tanh(FLAGS, last, 'GRU_out', False, last.get_shape()[-1], 512)
        print('GRU fc shape:', fc.shape)
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            fc = tf.layers.dropout(fc, rate=FLAGS.dropout, training=FLAGS.training)
        return fc



def BiGRU(FLAGS, inputs, BiGRU_scope, reuse_symbol):
    """
        inputs shape = [batch_size, lstm_time, cnn_out]
        max_time = left_context + 1(current_frame) + right_context
        define lstm
    """
    with tf.variable_scope(BiGRU_scope, reuse=reuse_symbol) as scope:
        if not reuse_symbol:
            inputs_hist = tf.summary.histogram('inputs', inputs)
        print("BiGRU_inputs shape:", inputs.shape)
        x = batch_norm_relu(FLAGS, inputs, 'BiGRU_inputs')
        gru_fw_cell = tf.contrib.rnn.GRUCell(FLAGS.lstm_hidden_units)
        gru_bw_cell = tf.contrib.rnn.GRUCell(FLAGS.lstm_hidden_units)
        try:
            outputs, _, _ = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, x, dtype=tf.float32)
        except Exception:
            outputs = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, x, dtype=tf.float32)
        output_fw, output_bw = outputs[-1]
        last = tf.concat([output_fw, output_bw], 1)
        last = tf.nn.relu(last)
        print('BiGRU outputs shape:', last.shape)
        fc = affine_layer_tanh(FLAGS, last, 'BiGRU_out', False, last.get_shape()[-1], 512)
        print('BiGRU fc shape:', fc.shape)
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            fc = tf.layers.dropout(fc, rate=FLAGS.dropout, training=FLAGS.training)
        return fc




def get_cnn_net3d(FLAGS, inputs, cnn_scope, reuse_symbol):
    """
    # inputs = [batch_size, lstm_time, neighbor_dim, feature_dim, 1]
    """
    with tf.variable_scope(cnn_scope, reuse=reuse_symbol) as scope:
        # CNN define
        num_inchannel = FLAGS.lstm_time / FLAGS.cnn_num_filter
        weights = {
            #'wc1': weight_variable([5, 5, FLAGS.lstm_time, 128], 'wc1'),
            'wc1': weight_variable([FLAGS.cnn_shift_time, 5, 5, 1, 128], 'wc1'),
            'wc2': weight_variable([1, 3, 3, 128, 256], 'wc2'),
            'wc3': weight_variable([1, 1, 4, 256, 512], 'wc3'),
#            'wd' : weight_variable([1 * 7 * 256, 1024], 'wd'),
        }

        biases = {
            'bc1': bias_variable([128], 'bc1'),
            'bc2': bias_variable([256], 'bc2'),
            'bc3': bias_variable([512], 'bc3'),
#            'bd' : bias_variable([1024], 'bd'),
        }

        strides = {
            'stride1': [1, FLAGS.cnn_shift_time, 2, 2, 1],
            'stride2': [1, 1, 1, 1, 1],
            'stride3': [1, 1, 1, 1, 1],
        }
        if not reuse_symbol:
            inputs_hist = tf.summary.histogram('inputs', inputs)
            wc1_hist = tf.summary.histogram('conv1/weights', weights['wc1'])
            bc1_hist = tf.summary.histogram('conv1/biases', biases['bc1'])
            wc2_hist = tf.summary.histogram('conv2/weights', weights['wc2'])
            bc2_hist = tf.summary.histogram('conv2/biases', biases['bc2'])
            wc3_hist = tf.summary.histogram('conv3/weights', weights['wc3'])
            bc3_hist = tf.summary.histogram('conv3/biases', biases['bc3'])
            #  wd_hist = tf.summary.histogram('cnn_fc/weights', weights['wd'])
            #  bd_hist = tf.summary.histogram('cnn_fc/biases', biases['bd'])

        #conv1
        tf.to_float(inputs)
        if not reuse_symbol:
            print("cnn inputs shape:", inputs.shape)
        #Couv-1
        conv1 = my_conv3d(FLAGS, inputs, weights['wc1'], biases['bc1'], 'conv3d1_layer', strides['stride1'])
        if FLAGS.batch_norm:
            conv1 = batch_norm_relu(FLAGS, conv1, 'conv1_bn')
        if not reuse_symbol:
            print("conv3d1 shape:", conv1.shape)
            conv1_hist = tf.summary.histogram('conv3d1_out', conv1)
        #max pool
        conv1 = tf.nn.max_pool3d(conv1, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name='max_pool3d1')
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            print("using conv1 dropout layer")
            conv1 = tf.layers.dropout(conv1, rate=FLAGS.dropout, training=FLAGS.training)
        if not reuse_symbol:
            conv1_maxpool_hist = tf.summary.histogram('conv3d1_pool_out', conv1)
            print("conv3d1 pool shape:", conv1.shape)
        #Conv-2
        conv2 = my_conv3d(FLAGS, conv1, weights['wc2'], biases['bc2'], 'conv3d2_layer')
        if FLAGS.batch_norm:
            conv2 = batch_norm_relu(FLAGS, conv2, 'conv2_bn')
        if not reuse_symbol:
            print("conv3d2 shape:", conv2.shape)
            conv2_hist = tf.summary.histogram('conv3d2_out', conv2)
        #max pool
        conv2 = tf.nn.max_pool3d(conv2, ksize=[1, 1, 1, 2, 1], strides=[1, 1, 1, 2, 1], padding='SAME', name='max_pool3d2')
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            print("using conv2 dropout layer")
            conv2 = tf.layers.dropout(conv2, rate=FLAGS.dropout, training=FLAGS.training)
        if not reuse_symbol:
            conv2_maxpool_hist = tf.summary.histogram('conv3d2_pool_out', conv2)
            print("conv3d2 pool shape:", conv2.shape)
        conv3 = my_conv3d(FLAGS, conv2, weights['wc3'], biases['bc3'], 'conv3d3_layer')
        if FLAGS.batch_norm:
            conv3 = batch_norm_relu(FLAGS, conv3, 'conv3_bn')
        print("conv3d3 shape:", conv3.shape)
        conv3 = tf.reshape(conv3, [-1, int(FLAGS.lstm_time / FLAGS.cnn_shift_time), 512])
        if not reuse_symbol:
            conv3_hist = tf.summary.histogram('conv3d3_out', conv3)
        return conv3


def get_lstm_net(FLAGS, inputs, lstm_scope, reuse_symbol):
    #inputs shape = [batch_size, lstm_time, cnn_out]
    #max_time = left_context + 1(current_frame) + right_context
    #define lstm
    with tf.variable_scope(lstm_scope, reuse=reuse_symbol) as scope:
        if not reuse_symbol:
            inputs_hist = tf.summary.histogram('inputs', inputs)
        tf.to_float(inputs)
        if not reuse_symbol:
            print("lstm inputs shape:", inputs.shape)
        lstm_cells = []
        for i in range(FLAGS.lstm_num_layers):
            lstm_cell = tf.contrib.rnn.GRUCell(FLAGS.lstm_hidden_units)
            #  if FLAGS.dropout_symbol and FLAGS.training:
            #      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=FLAGS.dropout)
            lstm_cells.append(lstm_cell)
        print("the number of hidden layers", FLAGS.lstm_num_layers)
        stack_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
        outputs, _ = tf.nn.dynamic_rnn(stack_lstm, inputs, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1,0,2])
        last = outputs[-1]
        last = PReLU(last, 'lstm_hidden_out')
        if FLAGS.batch_norm:
            last = batch_norm_prelu(FLAGS, last, 'lstm_hidden_output_bn')
            print("using batch norm: lstm_hidden_output_bn")
        if not reuse_symbol:
            print("lstm last shape:", last.shape)
            last_hist = tf.summary.histogram('lstm_hidden_out', last)
        fc = affine_layer_tanh(FLAGS, last, 'lstm_out', False, last.get_shape()[-1], 512)
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            fc = tf.layers.dropout(fc, rate=FLAGS.dropout, training=FLAGS.training)
        return fc

def prepare_model(inputs, num_classes, FLAGS):
    #inputs shape = [batch_size, lstm_time, neighbor_dim, feature_dim]
    batch_size = int(FLAGS.batch_size)
    lstm_time = int(FLAGS.lstm_time)
    neighbor_dim = int(FLAGS.left_context + FLAGS.right_context + 1)
    feature_dim = int(FLAGS.feature_dim)
    with tf.variable_scope('sre_cnn_net') as scope:
        print("inputs shape:", inputs.shape)
        cnn_inputs = tf.reshape(inputs, [-1, lstm_time, neighbor_dim, feature_dim, 1])
        cnn_outputs = get_cnn_net3d(FLAGS, cnn_inputs, 'cnn3d', False)
        print("cnn_outputs shape:", cnn_outputs.shape)
    with tf.variable_scope('sre_lstm_net') as scope:
        embeddings_w = weight_variable([FLAGS.dvector_dim, FLAGS.dvector_dim], 'embeddings_weights')
        embeddings_b = bias_variable([FLAGS.dvector_dim], 'embeddings_bias')
        weights = weight_variable([FLAGS.dvector_dim, num_classes], 'out_weights')
        biases = bias_variable([num_classes], 'out_biases')
        w_hist = tf.summary.histogram('dvector_out/weights', weights)
        b_hist = tf.summary.histogram('dvector_out/biases', biases)
        out = get_lstm_net(FLAGS, cnn_outputs, 'lstm', False)
        print("out shape:", out.shape)
        embeddings1 = out
        embeddings2 = tf.add(tf.matmul(embeddings1, embeddings_w), embeddings_b)
        if FLAGS.batch_norm:
            embeddings2 = batch_norm_tanh(FLAGS, embeddings2, 'embeddings2')
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            logits_inputs = tf.layers.dropout(embeddings2, rate=FLAGS.dropout, training=FLAGS.training)
        logits = tf.add(tf.matmul(embeddings2, weights), biases)
        print("logits shape:", logits.shape)
        logits_hist = tf.summary.histogram('logits', logits)
        return logits, embeddings1, embeddings2

def prepare_model_resnet(inputs, num_classes, FLAGS):
    #inputs shape = [batch_size, lstm_time, neighbor_dim, feature_dim]
    batch_size = int(FLAGS.batch_size)
    lstm_time = int(FLAGS.lstm_time)
    neighbor_dim = int(FLAGS.left_context + FLAGS.right_context + 1)
    feature_dim = int(FLAGS.feature_dim)
    with tf.variable_scope('res_net') as scope:
        print("inputs shape:", inputs.shape)
        resnet = get_resnet(FLAGS, inputs, 'resnet', False)
    with tf.variable_scope('sre_lstm_net') as scope:
        embeddings_w1 = weight_variable([512, FLAGS.dvector_dim], 'embeddings_weights1')
        embeddings_b1 = bias_variable([FLAGS.dvector_dim], 'embeddings_bias1')
        embeddings_w2 = weight_variable([FLAGS.dvector_dim, FLAGS.dvector_dim], 'embeddings_weights2')
        embeddings_b2 = bias_variable([FLAGS.dvector_dim], 'embeddings_bias2')
        weights = weight_variable([FLAGS.dvector_dim, num_classes], 'out_weights')
        biases = bias_variable([num_classes], 'out_biases')
        w_hist = tf.summary.histogram('dvector_out/weights', weights)
        b_hist = tf.summary.histogram('dvector_out/biases', biases)
        lstm_inputs = tf.reshape(resnet, [-1, FLAGS.lstm_time, 512])
        if FLAGS.batch_norm:
            lstm_inputs = batch_norm_prelu(FLAGS, lstm_inputs, 'lstm_inputs_bn')
        em1_inputs = get_lstm_net(FLAGS, lstm_inputs, 'lstm', False)
        embeddings1 = tf.add(tf.matmul(em1_inputs, embeddings_w1), embeddings_b1)
        embeddings1 = tf.tanh(embeddings1)
        if FLAGS.batch_norm:
            em2_inputs = batch_norm_tanh(FLAGS, embeddings1, 'embeddings1_bn')
        out1 = em2_inputs
        embeddings2 = tf.add(tf.matmul(em2_inputs, embeddings_w2), embeddings_b2)
        embeddings2 = tf.tanh(embeddings2)
        if FLAGS.batch_norm:
            logits_inputs = batch_norm_tanh(FLAGS, embeddings2, 'embeddings2_bn')
        out2 = logits_inputs
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            logits_inputs = tf.layers.dropout(logits_inputs, rate=FLAGS.dropout, training=FLAGS.training)
        logits = tf.add(tf.matmul(logits_inputs, weights), biases)
        print("logits shape:", logits.shape)
        logits_hist = tf.summary.histogram('logits', logits)
        #  return logits, embeddings1, embeddings2
        return logits, out1, out2

def prepare_model_BiGRU(inputs, num_classes, FLAGS):
    #inputs shape = [batch_size, lstm_time, neighbor_dim, feature_dim]
    batch_size = int(FLAGS.batch_size)
    lstm_time = int(FLAGS.lstm_time)
    neighbor_dim = int(FLAGS.left_context + FLAGS.right_context + 1)
    feature_dim = int(FLAGS.feature_dim)
    with tf.variable_scope('res_net') as scope:
        print("inputs shape:", inputs.shape)
        inputs = tf.reshape(inputs, [-1, lstm_time, neighbor_dim, FLAGS.feature_dim, 1])
        resnet = get_resnet(FLAGS, inputs, 'resnet', False)
    with tf.variable_scope('BiGRU_net') as scope:
        embeddings_w1 = weight_variable([512, FLAGS.dvector_dim], 'embeddings_weights1')
        embeddings_b1 = bias_variable([FLAGS.dvector_dim], 'embeddings_bias1')
        embeddings_w2 = weight_variable([FLAGS.dvector_dim, FLAGS.dvector_dim], 'embeddings_weights2')
        embeddings_b2 = bias_variable([FLAGS.dvector_dim], 'embeddings_bias2')
        weights = weight_variable([FLAGS.dvector_dim, num_classes], 'out_weights')
        biases = bias_variable([num_classes], 'out_biases')
        w_hist = tf.summary.histogram('dvector_out/weights', weights)
        b_hist = tf.summary.histogram('dvector_out/biases', biases)
        lstm_inputs = tf.reshape(resnet, [-1, FLAGS.lstm_time, 512])
        em1_inputs = BiGRU(FLAGS, lstm_inputs, 'BiGRU', False)
        embeddings1 = tf.add(tf.matmul(em1_inputs, embeddings_w1), embeddings_b1)
        embeddings1 = tf.tanh(embeddings1)
        if FLAGS.batch_norm:
            em2_inputs = batch_norm_tanh(FLAGS, embeddings1, 'embeddings1_bn')
        out1 = em2_inputs
        embeddings2 = tf.add(tf.matmul(em2_inputs, embeddings_w2), embeddings_b2)
        embeddings2 = tf.tanh(embeddings2)
        if FLAGS.batch_norm:
            logits_inputs = batch_norm_tanh(FLAGS, embeddings2, 'embeddings2_bn')
        out2 = logits_inputs
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            logits_inputs = tf.layers.dropout(logits_inputs, rate=FLAGS.dropout, training=FLAGS.training)
        logits = tf.add(tf.matmul(logits_inputs, weights), biases)
        print("logits shape:", logits.shape)
        logits_hist = tf.summary.histogram('logits', logits)
        #  return logits, embeddings1, embeddings2
        return logits, out1, out2

def prepare_model_res_GRU(inputs, num_classes, FLAGS):
    #inputs shape = [batch_size, lstm_time, neighbor_dim, feature_dim]
    batch_size = int(FLAGS.batch_size)
    lstm_time = int(FLAGS.lstm_time)
    neighbor_dim = int(FLAGS.left_context + FLAGS.right_context + 1)
    feature_dim = int(FLAGS.feature_dim)
    with tf.variable_scope('res_net') as scope:
        print("inputs shape:", inputs.shape)
        inputs = tf.reshape(inputs, [-1, lstm_time, neighbor_dim, FLAGS.feature_dim, 1])
        resnet = get_resnet(FLAGS, inputs, 'resnet', False)
    with tf.variable_scope('single_GRU_net') as scope:
        embeddings_w1 = weight_variable([512, FLAGS.dvector_dim], 'embeddings_weights1')
        embeddings_b1 = bias_variable([FLAGS.dvector_dim], 'embeddings_bias1')
        embeddings_w2 = weight_variable([FLAGS.dvector_dim, FLAGS.dvector_dim], 'embeddings_weights2')
        embeddings_b2 = bias_variable([FLAGS.dvector_dim], 'embeddings_bias2')
        weights = weight_variable([FLAGS.dvector_dim, num_classes], 'out_weights')
        biases = bias_variable([num_classes], 'out_biases')
        w_hist = tf.summary.histogram('dvector_out/weights', weights)
        b_hist = tf.summary.histogram('dvector_out/biases', biases)
        lstm_inputs = tf.reshape(resnet, [-1, FLAGS.lstm_time, 512])
        #  em1_inputs = single_GRU(FLAGS, lstm_inputs, 'single_GRU', False)
        em1_inputs = get_lstm_net(FLAGS, lstm_inputs, 'Multi_GRU', False)
        embeddings1 = tf.add(tf.matmul(em1_inputs, embeddings_w1), embeddings_b1)
        embeddings1 = tf.tanh(embeddings1)
        if FLAGS.batch_norm:
            em2_inputs = batch_norm_tanh(FLAGS, embeddings1, 'embeddings1_bn')
        out1 = em2_inputs
        embeddings2 = tf.add(tf.matmul(em2_inputs, embeddings_w2), embeddings_b2)
        embeddings2 = tf.tanh(embeddings2)
        if FLAGS.batch_norm:
            logits_inputs = batch_norm_tanh(FLAGS, embeddings2, 'embeddings2_bn')
        out2 = logits_inputs
        if FLAGS.dropout > 0 and FLAGS.dropout < 1:
            logits_inputs = tf.layers.dropout(logits_inputs, rate=FLAGS.dropout, training=FLAGS.training)
        logits = tf.add(tf.matmul(logits_inputs, weights), biases)
        print("logits shape:", logits.shape)
        logits_hist = tf.summary.histogram('logits', logits)
        #  return logits, embeddings1, embeddings2
        return logits, out1, out2
