#!/usr/bin/env python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import importlib
import os
import re
import sys

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
        with tf.io.gfile.GFile(frozen_graph,'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
    return graph


def detect_model_parameters(frozen_graph):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.io.gfile.GFile(frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='imported')
        ph = graph.get_tensor_by_name('imported/pointclouds:0')
        num_points = ph.get_shape()[1].value
        use_normal = (ph.get_shape()[2].value == 6)
    return num_points, use_normal


def build_knn_defense_network(graph, k=2, alpha=1.05):
    with graph.as_default():
        # Note that if #points is too large, it may cause OOM in this implementation,
        # since a large matrix will be created.
        points = tf.placeholder(tf.float32, shape=(None, 3), name='knn_points')
        k = tf.placeholder_with_default(2, shape=(), name='knn_k')
        alpha = tf.placeholder_with_default(alpha, shape=(), name='knn_alpha')

        num_points = tf.shape(points)[0]

        points_exp = tf.expand_dims(points, axis=1)
        points_exp = tf.tile(points_exp, multiples=[1, num_points, 1])

        distances = tf.sqrt(tf.reduce_sum(tf.square(points_exp - points), axis=-1))

        values, indices = tf.nn.top_k(tf.negative(distances), k=k+1)
        values = tf.negative(values[..., 1:])

        values = tf.reduce_mean(values, axis=-1)
        mean = tf.reduce_mean(values)
        stddev = tf.math.reduce_std(values)

        threshold = mean + alpha * stddev

        filtered = tf.math.less(values, threshold)
        filtered = tf.identity(filtered, name='knn_filtered')

    tensors = {
        'knn_points': points,
        'knn_k': k,
        'knn_alpha': alpha,
        'knn_filtered': filtered}

    return graph, tensors


def defense_random_shuffle(points):
    # try using python list to shuffle, then convert back to numpy array
    # use numpy shuffle will change data by a little amount, I don't know why
    pyl = points.tolist()
    ind = list(range(points.shape[0]))
    rd.shuffle(ind)

    shpyl = []
    for i in ind:
        shpyl.append(pyl[i])

    return np.array(shpyl)


def defense_knn_removal(sess, graph, tensors, points, k=2, alpha=1.05):
    assert points.ndim == 2
    mask = sess.run(tensors['knn_filtered'], feed_dict={
        tensors['knn_points']: points,
        tensors['knn_k']: k,
        tensors['knn_alpha']: alpha})
    mask = mask.astype(np.bool)

    return points[mask]


def inference(inputfiles, shape_names, model_path, num_votes, rounds, defenses):
    # Uncomment this line for more messages from Tensorflow
    tf.logging.set_verbosity(tf.logging.ERROR)

    normalize = True
    is_training = False

    num_classes = len(shape_names)

    shape2id = {}
    for idx, name in enumerate(shape_names):
        shape2id[name] = idx

    num_points, normal_channel = detect_model_parameters(model_path)
    assert normal_channel == False, 'Normal features are not available in the test sets.'

    graph = load_graph(model_path)

    with graph.as_default():
        pointclouds_pl = graph.get_tensor_by_name('pointclouds:0')
        is_training_pl = graph.get_tensor_by_name('is_training:0')
        preds = graph.get_tensor_by_name('fc3/BiasAdd:0')

    graph, knn_tensors = build_knn_defense_network(graph)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(graph=graph, config=config)

    ops = {'pointclouds_pl': pointclouds_pl,
           'is_training_pl': is_training_pl,
           'preds': preds}

    for fn in inputfiles:
        print('Processing file: {}'.format(fn))

        # only support point clouds (if meshes are loaded, assert will fail)
        point_set = meshio.loadmesh(fn)
        assert point_set.ndim == 2

        # Remove normals before any defense mechanisms are applied
        if not normal_channel:
            point_set = point_set[..., :3]
        if point_set.shape[0] > 10000:
            point_set = point_set[:10000, :]

        if 'knn' in defenses:
            knn_k = int(defenses['knn'][0])
            knn_alpha = float(defenses['knn'][1])
            point_set = defense_knn_removal(sess, graph, knn_tensors, point_set, k=knn_k, alpha=knn_alpha)

        if 'rr' in defenses:
            # Randomly shuffle points
            original_point_set = np.copy(point_set)
            point_set = []
            for i in range(rounds):
                point_set.append(defense_random_shuffle(original_point_set))
        else:
            point_set = [point_set] * rounds

        # Convert back to numpy array
        point_set = np.array(point_set)

        # Pick the first num_points for model input
        point_set = point_set[:, :num_points, :]

        # check the shape
        assert point_set.shape == (rounds, num_points, 3)

        if normalize:
            for i in range(rounds):
                point_set[i, ..., :3] = modelnet_dataset.pc_normalize(point_set[i, ..., :3])

        batch_pred_sum = np.zeros((rounds, num_classes))  # score for classes
        for vote_idx in range(num_votes):
            rotated_data = provider.rotate_point_cloud_by_angle(
                point_set, vote_idx / float(num_votes) * np.pi * 2)

            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['is_training_pl']: is_training}

            pred_val = sess.run(ops['preds'], feed_dict=feed_dict)
            batch_pred_sum += pred_val

        batch_pred_sum = extmath.softmax(batch_pred_sum)
        pred_id = np.argmax(batch_pred_sum, axis=1)
        assert batch_pred_sum.shape[0] == rounds
        assert pred_id.shape[0] == rounds

        for i in range(rounds):
            print('Predict class: {} with prob {:.4f}'.format(
                shape_names[pred_id[i]], batch_pred_sum[i][pred_id[i]]))


def main():
    parser = argparse.ArgumentParser(description='Inference PointNet++ with optional defense mechanisms')

    parser.add_argument('--model', type=str, default=None, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--rounds', type=int, default=10,
                        help='Number of inference, the results will be averaged')

    # Defense arguments
    parser.add_argument('--rr', action='store_true',
                        help='Randomly choose a subset from input point clouds')
    parser.add_argument('--knn', type=float, nargs=2, default=None,
                        help='Apply kNN neighbors outlier removal defense.'
                             'This option is used to specify the k and alpha value.')

    parser.add_argument('--num-votes', type=int, default=1,
                        help='Aggregate classification scores from multiple rotations')
    parser.add_argument('files', type=str, nargs='+',
                        help='Point cloud files to process')
    args = parser.parse_args()

    model_path = args.model
    num_votes = args.num_votes
    rounds = args.rounds

    if num_votes <= 0:
        raise ValueError('Invalid num_votes, should be greater than 0')

    defenses = dict()
    if args.rr:
        defenses['rr'] = True
    if args.knn is not None:
        defenses['knn'] = tuple(args.knn)
    if num_votes > 1:
        defenses['votes'] = num_votes

    shape_names = [line.rstrip() for line in
                   open('./labels/shape_names.txt')]

    inputfiles = argsutils.get_input_files(args.files)

    inference(inputfiles,
              shape_names,
              model_path,
              num_votes,
              rounds,
              defenses)


if __name__ == '__main__':
    main()
