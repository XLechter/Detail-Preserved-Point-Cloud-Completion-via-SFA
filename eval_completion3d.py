import argparse
import datetime
import importlib
# import models
import os
import tensorflow as tf
import time
from data_util import lmdb_dataflow, get_queued_data
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views
import numpy as np
import h5py


class TrainProvider:
    def __init__(self, args, is_training):
        df_train, self.num_train = lmdb_dataflow(args.lmdb_train, args.batch_size,
                                                 args.num_input_points, args.num_gt_points, is_training=True)
        batch_train = get_queued_data(df_train.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.num_input_points, 3],
                                       [args.batch_size, args.num_gt_points, 3]])
        df_valid, self.num_valid = lmdb_dataflow(args.lmdb_valid, args.batch_size,
                                                 args.num_input_points, args.num_gt_points, is_training=False)
        batch_valid = get_queued_data(df_valid.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.num_input_points, 3],
                                       [args.batch_size, args.num_input_points, 3]])
        self.batch_data = tf.cond(is_training, lambda: batch_train, lambda: batch_valid)


def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Note that theta is a parameter used for progressive training
    theta = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'theta_op')

    provider = TrainProvider(args, is_training_pl)
    ids, inputs, gt = provider.batch_data
    num_eval_steps = provider.num_valid // args.batch_size

    print('provider.num_valid', provider.num_valid)
    print('num_eval_steps', num_eval_steps)

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, gt, theta, False)
    add_train_summary('alpha', theta)

    # [new] to output pcds
    # out_path = '/mnt/data3/zwx/results_pcds'
    # f_out_pcd = h5py.File( os.path.join(out_path, 'SFA_point_comp_pcds.h5'), 'w')
    # g_output_pcd = f_out_pcd.create_group("output")

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
            synset_id = str(ids_eval[0]).split('_')[0].split('\'')[1]
            total_loss_fine += loss_fine
            total_time += time.time() - start

            if not cd_per_cat.get(synset_id):
                cd_per_cat[synset_id] = []
            cd_per_cat[synset_id].append(loss_fine)

            # [new] to output pcds
            # for i in range(args.batch_size):
            #     ind = args.batch_size * j + i
            #     g_output_pcd[f"{ind}"] = fine[i]

            #            print('ids : ', str(ids_eval[0]),synset_id)

            dir = str(ids_eval[0]).split('\'')[1].replace('_', '/')
            ofile = os.path.join('benchmark', dir + '.h5')
            if not os.path.exists('benchmark/all'):
                os.mkdir('benchmark')
                os.mkdir('benchmark/all')
            # fname = clouds_data[0][idx][:clouds_data[0][idx].rfind('.')]
            # synset = fname.split('/')[-2]
            # outp = outputs[idx:idx + 1, ...].squeeze()
            # odir = args.odir + '/benchmark/%s' % (synset)
            # if not os.path.isdir(odir):
            #     print("Creating %s ..." % (odir))
            #     os.makedirs(odir)
            # ofile = os.path.join(odir, fname.split('/')[-1])
            print("Saving to %s ..." % (ofile))
            print(fine.shape)
            with h5py.File(ofile, "w") as f:
                f.create_dataset("data", data=np.squeeze(fine))

            # if args.plot:
            #     for i in range(args.batch_size):
            #         model_id = str(ids_eval[i]).split('_')[1]
            #         os.makedirs(os.path.join(args.save_path, 'plots', synset_id), exist_ok=True)
            #         plot_path = os.path.join(args.save_path, 'plots', synset_id, '%s.png' % model_id)
            #         plot_pcd_three_views(plot_path, [inputs_eval[i], fine[i], gt_eval[i]],
            #                              ['input', 'output', 'ground truth'],
            #                              'CD %.4f' % (loss_fine),
            #                              [0.5, 0.5, 0.5])
        cur_dir = os.getcwd()
        subprocess.call("cd %s; zip -r submission.zip *; cd %s" % ('benchmark', cur_dir),
                        shell=True)
        print('zip file generated.')
        print('Average Chamfer distance: %f' % (total_loss_fine / num_eval_steps))
        # print(colored('epoch %d  step %d  loss %.8f loss_fine %.8f - time per batch %.4f' %
        #               (epoch, step, total_loss / num_eval_steps, total_loss_fine / num_eval_steps, total_time / num_eval_steps),
        #               'grey', 'on_green'))
        print('Chamfer distance per category')
        dict_known = {'02691156': 'airplane', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair',
                      '03636649': 'lamp', '04256520': 'sofa',
                      '04379243': 'table', '04530566': 'vessel'}
        dict_novel = {'02924116': 'Bus', '02818832': 'Bed', '02871439': 'bookshelf', '02828884': 'bench',
                      '03467517': 'guitar', '03790512': 'motorbike', '04225987': 'skateboard', '03948459': 'pistol'}
        dict_known_list = ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520',
                           '04379243', '04530566']
        dict_novel_list = ['02924116', '02818832', '02871439', '02828884',
                           '03467517', '03790512', '04225987', '03948459']
        temp_loss = 0
        for synset_id in dict_known_list:
            # print(len(dict_novel_list[:4]))
            temp_loss += np.mean(cd_per_cat[synset_id])
            print('%f' % np.mean(cd_per_cat[synset_id]), '&', end='')
        print(temp_loss / 8)
        # temp_loss=0
        # for synset_id in cd_per_cat.keys():
        #     temp_loss += np.mean(cd_per_cat[synset_id])
        #     print(dict[synset_id], '%f' % np.mean(cd_per_cat[synset_id]))
        break
    print('Total time', datetime.timedelta(seconds=time.time() - start_time))
    coord.request_stop()
    coord.join(threads)
    sess.close()

    f_out_pcd.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='/home/user/zwx/pcn/data/shapenet/train.lmdb')
    parser.add_argument('--lmdb_valid', default="/mnt/data3/zwx/test_completion3D.lmdb")
    parser.add_argument('--model_type', default='glfa')
    parser.add_argument('--model_path', default='/home/user/zwx/pcn3/SFA/trained_models/glfa')
    parser.add_argument('--save_path', default='data/rfa')
    parser.add_argument('--plot', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_input_points', type=int, default=2048)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    args = parser.parse_args()

    train(args)
