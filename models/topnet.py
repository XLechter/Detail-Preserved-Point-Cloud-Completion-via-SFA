# Code from TopNet

import tensorflow as tf
from tf_util import chamfer, add_train_summary, add_valid_summary
from net_util import mlp, mlp_conv
import os
import math
import numpy as np

tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]

def get_arch(nlevels, npts):
    logmult = int(math.log2(npts/2048))
    assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    while logmult > 0:
        last_min_pos = np.where(arch==np.min(arch))[0][-1]
        arch[last_min_pos]*=2
        logmult -= 1
    return arch

print(get_arch(8, 16384))

class Model:
    def __init__(self, inputs, gt, alpha):
        self.tarch = get_arch(6, 16384)
        self.features = self.create_encoder(inputs)
        self.fine = self.create_decoder(self.features)
        self.loss, self.loss_fine, self.update = self.create_loss(self.fine, self.fine, gt, alpha)
        self.outputs = self.fine
        self.visualize_ops = [inputs[0], self.fine[0], self.fine[0], gt[0]]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']

    def create_encoder(self, inputs):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
            print('features_global', features_global)
            features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
            print('features', features)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            print('features', features)
            features = tf.reduce_max(features, axis=1, name='maxpool_1')
            print('features', features)
        return features

    def create_level(self, level, input_channels, output_channels, inputs, bn):
        with tf.variable_scope('level_%d' % (level), reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [input_channels, int(input_channels/2),
                                         int(input_channels/4), int(input_channels/8),
                                         output_channels*int(self.tarch[level])],
                                         bn=bn)
            features = tf.reshape(features, [tf.shape(features)[0], -1, output_channels])
        return features

    def create_decoder(self, code):
        istraining = True
        NFEAT = 8
        Nin = NFEAT + 1024
        Nout = NFEAT
        bn = True
        print(self.tarch)
        N0 = int(self.tarch[0])
        nlevels = len(self.tarch)
        print('NFEAT * N0', NFEAT * N0)
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            level0 = mlp(code, [256, 64, NFEAT * N0], bn=bn)
            level0 = tf.tanh(level0, name='tanh_0')
            level0 = tf.reshape(level0, [-1, N0, NFEAT])
            outs = [level0, ]
            for i in range(1, nlevels):
                if i == nlevels - 1:
                    Nout = 3
                    bn = False
                inp = outs[-1]
                y = tf.expand_dims(code, 1)
                y = tf.tile(y, [1, tf.shape(inp)[1], 1])
                y = tf.concat([inp, y], 2)
                outs.append(tf.tanh(self.create_level(i, Nin, Nout, y, bn=bn), name='tanh_%d' % (i)))
        return outs[-1]

    def create_loss(self, coarse, fine, gt, alpha):

        loss_fine = chamfer(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        loss_coarse = loss_fine
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss = loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, loss_fine, [update_coarse, update_fine, update_loss]

