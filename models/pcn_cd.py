# Code from PCN

import tensorflow as tf
from tf_util import mlp, mlp_conv, chamfer, add_train_summary, add_valid_summary
import os

class Model:
    def __init__(self, inputs, gt, alpha):
        self.num_coarse = 1024
        self.grid_size = 4
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.features = self.create_encoder(inputs)
        self.coarse, self.fine = self.create_decoder(self.features)
        self.loss, self.loss_fine, self.update = self.create_loss(self.coarse, self.fine, gt, alpha)
        self.outputs = self.fine
        self.visualize_ops = [inputs[0], self.coarse[0], self.fine[0], gt[0]]
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

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])
            print('coarse', coarse)
        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size))
            print('grid:', grid)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            print('grid:', grid)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])
            print('grid_feat', grid_feat)

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])
            print('point_feat', point_feat)

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])

            print('global_feat', global_feat)

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)
            print('feat:', feat)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3])

            print('center', center)

            fine = mlp_conv(feat, [512, 512, 3]) + center
            print('fine:', fine)
        return coarse, fine

    def create_loss(self, coarse, fine, gt, alpha):
        loss_coarse = chamfer(coarse, gt)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_fine = chamfer(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        loss = loss_coarse + alpha * loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, loss_fine, [update_coarse, update_fine, update_loss]

