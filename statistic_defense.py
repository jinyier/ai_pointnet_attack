#!/usr/bin/env python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import importlib
import os
import re
import sys
import tqdm

import pickle as pkl
import random as rd

import numpy as np
import tensorflow as tf

import argsutils
import meshio
import modelnet_dataset

from sklearn.utils import extmath
from utils import provider

from tf_ops.grouping import tf_grouping
from tf_ops.interpolation_3d import tf_interpolate
from tf_ops.sampling import tf_sampling


def load_graph(frozen_graph):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph,'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
    return graph


def detect_model_parameters(frozen_graph):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='imported')
        ph = graph.get_tensor_by_name('imported/pointclouds:0')
        num_points = ph.get_shape()[1].value
        use_normal = (ph.get_shape()[2].value == 6)
    return num_points, use_normal


def defense_random_shuffle(points):
    pyl = points.tolist()
    ind = list(range(points.shape[0]))
    rd.shuffle(ind)

    shpyl = []
    for i in ind:
        shpyl.append(pyl[i])

    return np.array(shpyl)


def defense_gaussian_noise(points, alpha=1.0):
    assert points.ndim == 2
    noise = np.random.normal(loc=0, scale=alpha, size=points.shape)

    return points + noise


def get_groundtruth(fn, shape2id):
    fn = os.path.basename(fn).split('_')
    gt = fn[0]
    if gt in shape2id:
        return shape2id[gt]
    else:
        return shape2id['_'.join([fn[0], fn[1]])]


def inference(inputfiles, cleanfiles, shape_names, model_path, rounds, defenses, logpath):
    # Uncomment this line for more messages from Tensorflow
    tf.logging.set_verbosity(tf.logging.ERROR)

    logfile = logpath
    normalize = True
    is_training = False
    num_votes = 1
    has_threshold = (defenses['stat'][1] is not None)

    num_classes = len(shape_names)

    shape2id = {}
    for idx, name in enumerate(shape_names):
        shape2id[name] = idx

    num_points, normal_channel = detect_model_parameters(model_path)

    graph = load_graph(model_path)

    with graph.as_default():
        pointclouds_pl = graph.get_tensor_by_name('pointclouds:0')
        is_training_pl = graph.get_tensor_by_name('is_training:0')
        preds = graph.get_tensor_by_name('fc3/BiasAdd:0')

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(graph=graph, config=config)

    ops = {'pointclouds_pl': pointclouds_pl,
           'is_training_pl': is_training_pl,
           'preds': preds}

    print('Info: Sigma^2 for Gaussian Noise: {}'.format(defenses['stat']))

    # In this defense, experiments record all SIV values calculated (not success rate)
    experiments = dict()
    experiments['clean'] = list()
    experiments['adversarial'] = list()

    all_files = list()
    for fn in inputfiles:
        all_files.append((fn, False))
    if not has_threshold:
        # Only inference clean images if threshold is not given
        for fn in cleanfiles:
            all_files.append((fn, True))

    num_tasks = len(all_files)
    pbar = tqdm.tqdm(total=num_tasks, desc='Inference / Evaluate')

    for fn, is_clean in all_files:
        pbar.update(1)

        if is_clean:
            groundtruth_id = get_groundtruth(fn, shape2id)
        else:
            groundtruth_id = None

        # only support point clouds (if meshes are loaded, assert will fail)
        point_set = meshio.loadmesh(fn)
        assert point_set.ndim == 2

        # Remove normals before any defense mechanisms are applied
        if not normal_channel:
            point_set = point_set[..., :3]

        # Always randomly shuffle points
        point_set = np.array(defense_random_shuffle(point_set))

        # Pick the first num_points for model input
        # Do it here to reduce the calculation when applying Gaussian noise
        point_set = point_set[:num_points, :]

        # check the shape
        assert point_set.shape == (num_points, 3)

        unmodified_points = np.copy(point_set)

        # Do Gaussian Noise stuffs
        gn_alpha = defenses['stat'][0]
        original_point_set = np.copy(point_set)
        point_set = []
        for i in range(rounds):
            point_set.append(defense_gaussian_noise(original_point_set, alpha=gn_alpha))
        point_set = np.array(point_set)

        if normalize:
            unmodified_points = modelnet_dataset.pc_normalize(unmodified_points)
            for i in range(rounds):
                point_set[i, ..., :3] = modelnet_dataset.pc_normalize(point_set[i, ..., :3])

        pred_val = sess.run(ops['preds'], feed_dict={
            ops['pointclouds_pl']: [unmodified_points],
            ops['is_training_pl']: is_training})[0]
        assert pred_val.shape == (40,)

        batch_pred_sum = np.zeros((rounds, num_classes))  # score for classes
        for vote_idx in range(num_votes):
            rotated_data = provider.rotate_point_cloud_by_angle(
                point_set, vote_idx / float(num_votes) * np.pi * 2)

            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['is_training_pl']: is_training}

            pred_val = sess.run(ops['preds'], feed_dict=feed_dict)
            batch_pred_sum += pred_val

        batch_pred_sum = extmath.softmax(batch_pred_sum)
        pred_prob = np.copy(batch_pred_sum)
        pred_id = np.argmax(batch_pred_sum, axis=1)
        assert batch_pred_sum.shape == (rounds, num_classes)
        assert pred_prob.shape == (rounds, num_classes)
        assert pred_id.shape == (rounds,)

        var = np.var(pred_prob, axis=0)
        assert var.shape[0] == num_classes

        siv = np.mean(var)
        if is_clean:
            experiments['clean'].append(siv)
        else:
            experiments['adversarial'].append(siv)

    pbar.close()

    if defenses['stat'][1] is None:
        if not experiments['clean']:
            print('No Clean inputs found to estimate detection threshold')
            sys.exit(1)
        print('Use detection threshold from clean inference results')
        indices = np.argsort(np.array(experiments['clean']))[::-1]
        num_examples = indices.shape[0]
        detect_threshold_easy = experiments['clean'][indices[int(num_examples * 0.05)]]
        detect_threshold_hard = experiments['clean'][indices[int(num_examples * 0.1)]]
    else:
        print('Use detection threshold from commandline arguments')
        detect_threshold_easy = float(defenses['stat'][1])
        detect_threshold_hard = float(defenses['stat'][1])

    def print_eval_result(fid):
        if has_threshold:
            print('Detection Threshold: {}'.format((detect_threshold_hard)))
        else:
            print('Detection Threshold: {}'.format((detect_threshold_easy, detect_threshold_hard)))
        siv = np.array(experiments['adversarial'])
        print('Average: {}'.format(siv.mean()), file=fid)
        print('    Detected: {} / {} ({:.4f})'.format(
            np.sum(siv > detect_threshold_easy), siv.shape[0],
            np.sum(siv > detect_threshold_easy) / siv.shape[0]),
            file=fid)
        print('    Detected: {} / {} ({:.4f})'.format(
            np.sum(siv > detect_threshold_hard), siv.shape[0],
            np.sum(siv > detect_threshold_hard) / siv.shape[0]),
            file=fid)

    print_eval_result(sys.stdout)


def main():
    parser = argparse.ArgumentParser(description='Inference PointNet++ with Gaussian Random Noise')

    # Core arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use')
    parser.add_argument('--model', type=str, default=None, required=False,
                        help='Path to model frozen graph')

    # Defense arguments
    parser.add_argument('--stat', type=float, default=None,
                        help='Apply statistic defense, specify sigma^2 for random Gaussian Noise')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Detection threshold for SIVs')

    # Inference arguments
    parser.add_argument('--log', type=str, default=None,
                        help='Path to log file')
    parser.add_argument('--clean', type=str, default='./data/modelnet40_normal_resampled/',
                        help='Path to clean images')
    parser.add_argument('files', type=str, nargs='+',
                        help='Point files to process (txt, asc, xyz)')
    args = parser.parse_args()

    gpu_index = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    model_path = args.model
    logpath = args.log
    rounds = 10

    defenses = dict()
    defenses['stat'] = (args.stat,) + (args.threshold,)

    shape_names = [line.rstrip() for line in
                   open('./labels/shape_names.txt')]

    cleandirs = os.listdir(args.clean)
    cleandirs = [os.path.join(args.clean, dirname) for dirname in cleandirs]
    cleandirs = [dirname for dirname in cleandirs
                 if os.path.isdir(dirname)]

    inputfiles = argsutils.get_input_files(args.files)
    cleanfiles = argsutils.get_input_files(cleandirs)

    inference(inputfiles,
              cleanfiles,
              shape_names,
              model_path,
              rounds,
              defenses,
              logpath)


if __name__ == '__main__':
    main()
