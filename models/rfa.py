
import tensorflow as tf
from tf_util import mlp, mlp_conv, chamfer, add_train_summary, add_valid_summary, earth_mover
from utils import tf_util2
from utils.pointnet_util import pointnet_sa_module,pointnet_fp_module
import os
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from tf_ops.sampling import tf_sampling
from tf_ops.grouping.tf_grouping import query_ball_point, group_point
from transform_nets import input_transform_net, feature_transform_net

class Model:
    def __init__(self, inputs, gt, theta, is_training):
        self.num_coarse = 4096
        self.grid_size = 2
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.coarse_highres, self.coarse, self.fine = self.completion(inputs, is_training)
        self.loss, self.loss_fine, self.update = self.create_loss(self.coarse_highres, self.coarse, self.fine, gt, theta)
        self.outputs = self.fine
        self.visualize_ops = [inputs[0], self.coarse[0], self.fine[0], gt[0]]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']


    def completion(self, inputs, is_training):
        num_point = inputs.get_shape()[1].value
        l0_xyz = inputs[:,:,0:3]
        l0_points = None

        is_training = is_training
        bradius = 1.0
        use_bn = False
        use_ibn = False
        bn_decay = 0.95
        up_ratio = 8

        self.grid_size = 2
        self.num_coarse = int(num_point * up_ratio / 4)

        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=num_point,
                                                               radius=bradius * 0.05, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer1')

            l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point / 2,
                                                               radius=bradius * 0.1, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[64, 64, 128], mlp2=None,
                                                               group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer2')

            l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point / 4,
                                                               radius=bradius * 0.2, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[128, 128, 256], mlp2=None,
                                                               group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer3')

            l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point / 8,
                                                               radius=bradius * 0.3, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[256, 256, 512], mlp2=None,
                                                               group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer4')

            l5_xyz, l5_points, l5_indices = pointnet_sa_module(l4_xyz, l4_points, npoint=num_point / 16,
                                                               radius=bradius * 0.4, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[512, 512, 1024], mlp2=None,
                                                               group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer5')

            gl_xyz, gl_points, gl_indices = pointnet_sa_module(l5_xyz, l5_points, npoint=1,
                                                               radius=bradius * 0.3, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[512, 512, 1024], mlp2=None,
                                                               group_all=True,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer6')

            gl_feature = tf.reduce_max(gl_points, axis=1)

            print('gl_feature', gl_feature)

            # Feature Propagation layers

            up_gl_points = pointnet_fp_module(l0_xyz, gl_xyz, None, gl_points, [64], is_training, bn_decay,
                                              scope='fa_layer0', bn=use_bn, ibn=use_ibn)

            up_l5_points = pointnet_fp_module(l0_xyz, l5_xyz, None, l5_points, [64], is_training, bn_decay,
                                              scope='fa_layer1', bn=use_bn, ibn=use_ibn)

            up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
                                              scope='fa_layer2', bn=use_bn, ibn=use_ibn)

            up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
                                              scope='fa_layer3', bn=use_bn, ibn=use_ibn)

            up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay,
                                              scope='fa_layer4', bn=use_bn, ibn=use_ibn)

            ###concat feature
        with tf.variable_scope('up_layer', reuse=tf.AUTO_REUSE):
            new_points_list = []
            for i in range(up_ratio):
                if i>3:
                    transform = input_transform_net(l0_xyz, is_training, bn_decay, K=3)
                    xyz_transformed = tf.matmul(l0_xyz, transform)

                    concat_feat = tf.concat([up_gl_points, up_gl_points-up_l5_points, up_gl_points-up_l4_points, up_gl_points-up_l3_points, up_gl_points-up_l2_points, up_gl_points-l1_points, xyz_transformed],
                                            axis=-1)
                    print('concat_feat1', concat_feat)
                else:
                    concat_feat = tf.concat([up_gl_points, up_l5_points, up_l4_points, up_l3_points, up_l2_points, l1_points, l0_xyz],
                                            axis=-1)
                    print('concat_feat2', concat_feat)
                #concat_feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points, l0_xyz], axis=-1)
                concat_feat = tf.expand_dims(concat_feat, axis=2)
                concat_feat = tf_util2.conv2d(concat_feat, 256, [1, 1],
                                              padding='VALID', stride=[1, 1],
                                              bn=False, is_training=is_training,
                                              scope='fc_layer0_%d' % (i), bn_decay=bn_decay)

                new_points = tf_util2.conv2d(concat_feat, 128, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=use_bn, is_training=is_training,
                                             scope='conv_%d' % (i),
                                             bn_decay=bn_decay)
                new_points_list.append(new_points)
            net = tf.concat(new_points_list, axis=1)

            coord_feature = tf_util2.conv2d(net, 64, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=False, is_training=is_training,
                                    scope='fc_layer1', bn_decay=bn_decay)


            coord = tf_util2.conv2d(coord_feature, 3, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=False, is_training=is_training,
                                    scope='fc_layer2', bn_decay=bn_decay,
                                    activation_fn=None, weight_decay=0.0)  # B*(2N)*1*3

            coarse_highres = tf.squeeze(coord, [2])  # B*(2N)*3
            coord_feature = tf.squeeze(coord_feature, [2])
            fps_idx = farthest_point_sample(int(self.num_fine)/2, coarse_highres)
            coord_feature = gather_point(coord_feature, fps_idx)
            coarse_fps = gather_point(coarse_highres, fps_idx)

            coord_feature = tf.expand_dims(coord_feature, 2)

            print('coord_feature', coord, coord_feature)

            score = tf_util2.conv2d(coord_feature, 16, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=False, is_training=is_training,
                                    scope='fc_layer3', bn_decay=bn_decay)

            score = tf_util2.conv2d(score, 8, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=False, is_training=is_training,
                                    scope='fc_layer4', bn_decay=bn_decay)

            score = tf_util2.conv2d(score, 1, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=False, is_training=is_training,
                                    scope='fc_layer5', bn_decay=bn_decay)

            score = tf.nn.softplus(score)
            score = tf.squeeze(score, [2,3])

            _, idx = tf.math.top_k(score, self.num_coarse)

            coarse = gather_point(coarse_fps, idx)

            coord_feature = tf.squeeze(coord_feature, [2])
            coord_feature = gather_point(coord_feature, idx)

            print('coarse', coord_feature, coarse)


        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size))
            print('grid:', grid)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            print('grid:', grid)
            grid_feat = tf.tile(grid, [coarse.shape[0], self.num_coarse, 1])
            print('grid_feat', grid_feat)

            point_feat = tf.tile(tf.expand_dims(tf.concat([coarse, coord_feature], axis=-1), 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [coarse.shape[0], self.num_fine, -1])
            print('point_feat', point_feat)

            global_feat = tf.tile(tf.expand_dims(gl_feature, 1), [1, self.num_fine, 1])

            #print('global_feat', global_feat)

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)
            print('feat:', feat)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3])

            print('center', center)

            fine = mlp_conv(feat, [512, 512, 3]) + center
            print('fine:', fine)

        return coarse_highres, coarse, fine

    def create_loss(self, coarse_highres, coarse, fine, gt, theta):
        loss_coarse_highres = chamfer(coarse_highres, gt)

        loss_coarse = chamfer(coarse, gt)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_fine = chamfer(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        repulsion_loss = get_repulsion_loss4(coarse)

        loss = 0.5 * loss_coarse_highres + loss_coarse + theta * loss_fine + 0.2 * repulsion_loss
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, loss_fine, [update_coarse, update_fine, update_loss]

def get_repulsion_loss4(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12, dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square / h ** 2)
    uniform_loss = tf.reduce_mean(radius - dist * weight)
    return uniform_loss
