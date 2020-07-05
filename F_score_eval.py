import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
from data_util import lmdb_dataflow, get_queued_data
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views
import numpy as np
import open3d
import typing

def calculate_fscore(gt: open3d.geometry.PointCloud, pr: open3d.geometry.PointCloud, th: float = 0.01) -> typing.Tuple[
    float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pr)
    d2 = open3d.compute_point_cloud_to_point_cloud_distance(pr, gt)

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0

    return fscore

class TrainProvider:
    def __init__(self, args, is_training):
        df_test, self.num_test = lmdb_dataflow(args.lmdb_test, args.batch_size,
                                                 args.num_input_points, args.num_gt_points, is_training=False)
        batch_test = get_queued_data(df_test.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.num_input_points, 3],
                                       [args.batch_size, args.num_gt_points, 3]])
        self.batch_data = batch_test


def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    #Note that theta is a parameter used for progressive training
    theta = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'theta_op')

    provider = TrainProvider(args, is_training_pl)
    ids, inputs, gt = provider.batch_data
    num_eval_steps = provider.num_test // args.batch_size

    print('provider.num_valid', provider.num_test)
    print('num_eval_steps', num_eval_steps)

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, gt, theta, False)
    add_train_summary('alpha', theta)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=10)
    saver.restore(sess, args.model_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    start_time = time.time()
    while not coord.should_stop():
        print(colored('Testing...', 'grey', 'on_green'))
        total_time = 0
        total_loss_fine = 0
        cd_per_cat = {}
        sess.run(tf.local_variables_initializer())
        for j in range(num_eval_steps):
            start = time.time()
            ids_eval, inputs_eval, gt_eval, loss_fine, fine = sess.run([ids, inputs, gt, model.loss_fine, model.fine],
                               feed_dict={is_training_pl: False})
            pc_gt = open3d.geometry.PointCloud()
            pc_pr = open3d.geometry.PointCloud()
            print(np.squeeze(gt_eval).shape)
            pc_gt.points = open3d.Vector3dVector(np.squeeze(gt_eval))
            pc_pr.points = open3d.Vector3dVector(np.squeeze(fine))

            f_score = calculate_fscore(pc_gt, pc_pr)
            # print('f_score:', f_score)

            synset_id = str(ids_eval[0]).split('_')[0].split('\'')[1]
            total_loss_fine += f_score
            total_time += time.time() - start

            if not cd_per_cat.get(synset_id):
                cd_per_cat[synset_id] = []
            cd_per_cat[synset_id].append(f_score)

            # if args.plot:
            #     for i in range(args.batch_size):
            #         model_id = str(ids_eval[i]).split('_')[1]
            #         os.makedirs(os.path.join(args.save_path, 'plots', synset_id), exist_ok=True)
            #         plot_path = os.path.join(args.save_path, 'plots', synset_id, '%s.png' % model_id)
            #         plot_pcd_three_views(plot_path, [inputs_eval[i], fine[i], gt_eval[i]],
            #                              ['input', 'output', 'ground truth'],
            #                              'CD %.4f' % (loss_fine),
            #                              [0.5, 0.5, 0.5])
        print('Average F_score: %f' % (total_loss_fine / num_eval_steps))
        print('F_score per category')
        dict_known = {'02691156': 'airplane','02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03636649': 'lamp', '04256520': 'sofa',
                '04379243' : 'table','04530566': 'vessel'}
        temp_loss=0
        for synset_id in dict_known.keys():
            temp_loss += np.mean(cd_per_cat[synset_id])
            print(dict_known[synset_id], ' %f' % np.mean(cd_per_cat[synset_id]))
        break
    print('Total time', datetime.timedelta(seconds=time.time() - start_time))
    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_test', default='data/shapenet/test.lmdb')
    parser.add_argument('--model_type', default='rfa')
    parser.add_argument('--model_path', default='data/trained_models/rfa')
    parser.add_argument('--save_path', default='data/rfa')
    parser.add_argument('--plot', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_input_points', type=int, default=2048)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    args = parser.parse_args()

    train(args)