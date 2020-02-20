#!/usr/bin/env python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import importlib
import logging
import os
import re
import sys

import numpy as np
import random as rd
import tensorflow as tf

import argsutils
import meshio
import modelnet_dataset

from tf_ops.grouping import tf_grouping
from tf_ops.interpolation_3d import tf_interpolate
from tf_ops.sampling import tf_sampling


def load_graph(frozen_graph, raw_num_points, num_points, random_distortions):
    graph = tf.Graph()
    with graph.as_default():
        # Input placeholders
        # Point cloud (Point set)
        # The placeholder has no batch axis since we only deal with a pointcloud in each attack process
        pointcloud_pl = tf.placeholder(tf.float32, shape=(raw_num_points, 3), name='pointcloud_orig')
        # Normal vector of point clouds
        pointnormal_pl = tf.placeholder(tf.float32, shape=(raw_num_points, 3), name='pointnormal_orig')
        # Rotation angle in degree [0, 360), used for random distortion
        rotate_matrix_pl = tf.placeholder(tf.float32, shape=(None, 3, 3), name='rotate_matrix')
        # Used for batch normalization layer
        is_training_pl = tf.placeholder_with_default(False, shape=(), name='is_training_default')

        # L0 mask for perturbation
        l0_mask_pl = tf.placeholder(tf.float32, shape=(raw_num_points), name='l0_mask')
        l0_mask = tf.stack([l0_mask_pl] * 3, axis=-1)

        # Variable to optimize
        perturb = tf.Variable(np.zeros((raw_num_points, 3)), dtype=tf.float32, name='perturb')

        # l0 masked perturbation
        perturb_masked = perturb * l0_mask

        # Modified point clouds
        pointcloud = tf.math.add(pointcloud_pl, perturb_masked)

        # Output of adversarial pointclouds
        assert pointcloud.shape[0].value == raw_num_points
        pointcloud_output = tf.identity(pointcloud, name='pointcloud_pert')

        # Random sample for model input
        assert pointcloud.shape[0].value == raw_num_points
        pointcloud_sampled = random_sample_pointcloud(pointcloud, num_samples=num_points)
        pointcloud_sampled = tf.identity(pointcloud_sampled, name='pointcloud_sampled')

        # Random sample for knn distance
        assert pointcloud.shape[0].value == raw_num_points
        pointcloud_knn_sampled = tf.identity(pointcloud_sampled, name='pointcloud_knn_sampled')

        # Normalize
        assert pointcloud_sampled.shape[0].value == num_points
        pointcloud = normalize_pointcloud(pointcloud_sampled)

        if random_distortions > 0:
            batch_size = tf.shape(rotate_matrix_pl)[0]
            pointclouds = tf.broadcast_to(pointcloud, shape=(batch_size, num_points, 3))
            pointclouds = tf.linalg.matmul(pointclouds, rotate_matrix_pl)
        else:
            pointclouds = tf.expand_dims(pointcloud, axis=0)
        assert pointclouds.shape[1].value == num_points

        graphdef = tf.GraphDef()
        with tf.io.gfile.GFile(frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            graphdef.ParseFromString(serialized_graph)
            tf.import_graph_def(graphdef, name='', input_map={
                                'pointclouds': pointclouds,
                                'is_training': is_training_pl})

    with graph.as_default():
        feats = graph.get_tensor_by_name('fc2/BiasAdd:0')
        feats = tf.identity(feats, name='feats')
        logits = graph.get_tensor_by_name('fc3/BiasAdd:0')
        logits = tf.identity(logits, name='logits')
        probs = tf.nn.softmax(logits, axis=-1, name='probs')

    with graph.as_default():
        tensors = {'pc_orig': graph.get_tensor_by_name('pointcloud_orig:0'),
                   'pc_pert': graph.get_tensor_by_name('pointcloud_pert:0'),
                   'pc_samp': graph.get_tensor_by_name('pointcloud_sampled:0'),
                   'knn_samp': graph.get_tensor_by_name('pointcloud_knn_sampled:0'),
                   'nv_orig': graph.get_tensor_by_name('pointnormal_orig:0'),
                   'rot_mat': graph.get_tensor_by_name('rotate_matrix:0'),
                   'l0_mask': graph.get_tensor_by_name('l0_mask:0'),
                   'logits': graph.get_tensor_by_name('logits:0'),
                   'probs': graph.get_tensor_by_name('probs:0'),
                   'feats': graph.get_tensor_by_name('feats:0'),
                   'pert': perturb}

    return graph, tensors


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


def load_pointcloud(fn, shape2id, logger):
    logger.info('Reading file {}...'.format(fn))
    bn = os.path.basename(fn)
    shapename = re.split('[._]', bn)[0]
    groundtruth = shape2id[shapename]
    logger.info('  Groundtruth: {}'.format(shapename))

    pointcloud = meshio.loadmesh(fn)

    pointnormal = pointcloud[..., 3:]
    pointcloud = pointcloud[..., :3]

    if pointnormal.size == 0:
        logger.warning('  Warning: Input data has no normal information')
        logger.warning('        -> Fill them with all zeros')
        pointnormal = np.zeros_like(pointcloud)

    return groundtruth, pointcloud, pointnormal


def random_sample_and_add_points(points, normals, raw_num_points, logger):
    # raw_num_points: final number of points
    num_points = points.shape[0]
    num_adds = raw_num_points - num_points

    if num_adds > 0:
        logger.info('  Expected #points > current #points. Points added: {}'.format(num_adds))
        round_add = num_adds // num_points + 1
        points = np.concatenate([points] * round_add, axis=0)
        normals = np.concatenate([normals] * round_add, axis=0)

        ind_fixed = np.arange(num_points)
        ind_range = np.arange(points.shape[0], dtype=np.int32)
        np.random.shuffle(ind_range)

        ind_range = np.concatenate([ind_fixed, ind_range], axis=0)
        points = points[ind_range[:raw_num_points]]
        normals = normals[ind_range[:raw_num_points]]

    else:
        logger.info('  Expected #points <= current #points. Choose a subset')
        ind_range = np.arange(points.shape[0], dtype=np.int32)
        np.random.shuffle(ind_range)

        points = points[ind_range[:raw_num_points]]
        normals = normals[ind_range[:raw_num_points]]

    return points, normals


def random_sample_pointcloud(points, num_samples):
    # points: shape n x 3, where n is num_points
    num_pts = points.get_shape()[0].value
    ind = tf.range(num_pts)
    rind = tf.random_shuffle(ind)[:num_samples]
    rpoints = tf.gather(points, rind, axis=0)
    return rpoints


def normalize_pointcloud(points):
    # points: shape n x 3, where n is num_points
    # tf implementation of modelnet_dataset.pc_normalize()
    num_pts = points.get_shape()[0].value
    centroid = tf.reduce_mean(points, axis=0)
    points = points - centroid
    max_pts = tf.reduce_max(tf.reduce_sum(tf.square(points), axis=1))
    points = tf.math.divide(points, max_pts)
    return points


def chamfer_distance(points_x, points_y):
    # chamfer distance from point set x to point set y
    # x will be stack to have shape [#pt_x, #pt_y, 3]
    num_points = tf.shape(points_y)[0]
    points_x = tf.expand_dims(points_x, axis=1)
    points_x = tf.tile(points_x, multiples=[1, num_points, 1])
    chamfer = tf.square(points_x - points_y)    # n x n x 3
    chamfer = tf.reduce_sum(chamfer, axis=-1)   # n x n
    chamfer = tf.reduce_min(chamfer, axis=-1)   # n
    chamfer = tf.reduce_sum(chamfer, axis=-1)   # 1
    return chamfer


def knn_outlier_distance(points, points_all, k=5, alpha=1.05):
    # points: shape n x 3, where n is num_points
    num_points = points_all.shape[0].value

    points_now = tf.expand_dims(points, axis=1)
    points_now = tf.tile(points_now, multiples=[1, num_points, 1])
    distance = tf.square(points_now - points_all)        # n x n x 3
    distance = tf.reduce_sum(distance, axis=-1)          # n x n

    values, indices = tf.nn.top_k(tf.negative(distance), k=k+1)
    values, indices = values[..., 1:], indices[..., 1:]  # n x k
    values = tf.negative(values)

    avg_distance = tf.reduce_mean(values, axis=-1)       # n
    knn_mean = tf.reduce_mean(avg_distance)              # 1
    knn_stddev = tf.math.reduce_std(avg_distance)        # 1
    threshold = knn_mean + alpha * knn_stddev

    condition = tf.math.greater_equal(avg_distance, threshold)
    penalty = tf.where(condition, avg_distance, tf.zeros_like(avg_distance))
    penalty = tf.reduce_sum(penalty)
    return penalty


def gradients_clipping(gradvars, normals):
    # gradvars: a list returned by tf.train.Optimizer.compute_gradients()
    # normals: shape n x 3, normal vector of the object
    assert len(gradvars) == 1   # w.r.t. perturbation

    gradvalue = gradvars[0][0]
    gradname = gradvars[0][1]
    inner_prod = tf.reduce_sum(tf.multiply(tf.negative(gradvalue), normals))
    preserved = tf.math.greater_equal(inner_prod, tf.constant(0.0))
    gradvalue = tf.where(preserved, gradvalue, tf.zeros_like(gradvalue))

    return [(gradvalue, gradname)]


def generate_random_rotations(batch_size):
    degs = []
    mats = []
    for i in range(batch_size):
        degs.append(rd.randrange(0, 360))
    for deg in degs:
        rad = np.deg2rad(deg)
        cosval = np.cos(rad)
        sinval = np.sin(rad)
        mats.append([
            [ cosval,   0.0, sinval],
            [    0.0,   1.0,    0.0],
            [-sinval,   0.0, cosval]])
    return np.array(mats)


def build_perturbation_clipping_network(graph, tensors, project='dir'):
    if project not in ['dir', 'norm', 'none']:
        raise ValueError('Invalid projection type: {}'.format(project))

    with graph.as_default():
        cc_linf_pl = tf.placeholder(tf.float32, shape=(), name='cc_linf')
        tensors['cc_linf'] = cc_linf_pl

        normal = tensors['nv_orig']
        perturb = tensors['pert']

        # compute inner product
        inner_prod = tf.reduce_sum(normal * perturb, axis=-1)  # shape: n
        condition_inner = tf.math.greater_equal(inner_prod, tf.constant(0.0))

        if project == 'dir':
            # 1) vng = Normal x Perturb
            # 2) vref = vng x Normal
            # 3) Project Perturb onto vref
            #    Note that the length of vref should be greater than zero
            vng = tf.linalg.cross(normal, perturb)
            vng_len = tf.sqrt(tf.reduce_sum(tf.square(vng), axis=-1))

            vref = tf.linalg.cross(vng, normal)
            vref_len = tf.sqrt(tf.reduce_sum(tf.square(vref), axis=-1))
            vref_len_stack = tf.stack([vref_len] * 3, axis=-1)

            # add 1e-6 to avoid dividing by zero
            perturb_projected = perturb * vref / (vref_len_stack + 1e-6)

            # if the length of vng < 1e-6, let projected vector = (0, 0, 0)
            # it means the Normal and Perturb are just in opposite direction
            condition_vng = tf.math.greater(vng_len, tf.constant(1e-6))
            perturb_projected = tf.where(condition_vng, perturb_projected, tf.zeros_like(perturb_projected))

            # if inner_prod < 0, let perturb be the projected ones
            perturb = tf.where(condition_inner, perturb, perturb_projected)

        elif project == 'norm':
            # 1) Project Perturb onto normal
            # 2) Choose based on inner product
            normal_len = tf.sqrt(tf.reduce_sum(tf.square(normal), axis=-1))
            normal_len_stacked = tf.stack([normal_len] * 3, axis=-1)

            # the length of normal vector should always be one
            perturb_projected = perturb * normal / (normal_len_stacked + 1e-6)

            # if inner_prod < 0, let perturb be the projected ones
            perturb = tf.where(condition_inner, perturb_projected, tf.zeros_like(perturb_projected))

        else:
            # without projection, let the perturb be (0, 0, 0) if inner_prod < 0
            perturb = tf.where(condition_inner, perturb, tf.zeros_like(perturb))

        # compute vector length
        # if length > cc_linf, clip it
        lengths = tf.sqrt(tf.reduce_sum(tf.square(perturb), axis=-1))
        lengths_stacked = tf.stack([lengths] * 3, axis=-1)     # shape: n x 3

        # scale the perturbation vectors to length cc_linf
        # except the ones with zero length
        condition = tf.math.greater(lengths, tf.constant(1e-6))
        perturb_scaled = tf.where(condition, perturb / lengths_stacked * cc_linf_pl, tf.zeros_like(perturb))

        # check the length and clip if necessary
        condition = tf.math.less_equal(lengths, cc_linf_pl)
        perturb = tf.where(condition, perturb, perturb_scaled)

        # assign operatior for updating the perturbation variable
        perturb_assign = tf.assign(tensors['pert'], perturb)
        tensors['pert_assign'] = perturb_assign

    return graph, tensors


def build_normal_estimate_network(graph, tensors, k=3):
    # k: number of neighbors used in kNN algorithm
    with graph.as_default():
        # Note that the first dimension should be the same between points_orig and normals_orig
        points_pert = tf.placeholder(tf.float32, shape=(3), name='points_pert_single')
        points_orig = tensors['pc_orig']
        normals_orig = tensors['nv_orig']

        distance = tf.square(points_orig - points_pert)
        distance = tf.reduce_sum(distance, axis=-1)

        values, indices = tf.nn.top_k(tf.negative(distance), k=k)
        values = tf.negative(values)                           # k

        normals_top1 = tf.gather(normals_orig, indices[0])     # 3
        avg1_normals = tf.identity(normals_top1)

        normals_topk = tf.gather(normals_orig, indices)        # k x 3
        avgk_normals = tf.reduce_mean(normals_topk, axis=0)    # 3
        avgk_lengths = tf.sqrt(tf.reduce_sum(tf.square(avgk_normals), axis=-1))
        avgk_lengths = tf.stack([avgk_lengths] * 3, axis=-1)
        avgk_normals = tf.divide(avgk_normals, avgk_lengths)   # 3  (normalize the vector)

        # If the points are not modified (distance = 0), use the normal directly from the original
        # one. Otherwise, use the mean of the normals of the k-nearest points.
        exact = tf.math.less(tf.math.abs(values[0]), tf.constant(1e-6))
        normals_pert = tf.where(exact, avg1_normals, avgk_normals)
        normals_pert = tf.identity(normals_pert, 'pointnormal_pert')

        tensors['pc_pert_single'] = points_pert
        tensors['nv_pert'] = normals_pert

    return graph, tensors


def build_knn_centroid_network(graph, tensors, k=5):
    with graph.as_default():
        points_inp = tf.placeholder(tf.float32, shape=(None, 3), name='knn_res')
        points_ref = tf.placeholder(tf.float32, shape=(None, 3), name='knn_ref')

        num_inp = tf.shape(points_inp)[0]
        num_ref = tf.shape(points_ref)[0]

        tiled_inp = tf.expand_dims(points_inp, axis=1)
        tiled_inp = tf.tile(tiled_inp, multiples=[1, num_ref, 1])

        distance = tf.square(tiled_inp - points_ref)    # np x no x 3
        distance = tf.reduce_sum(distance, axis=-1)     # np x no

        values, indices = tf.nn.top_k(tf.negative(distance), k=k)

        ref_points = tf.gather(points_ref, indices)
        ref_points = tf.reduce_mean(ref_points, axis=1)

        ref_points = tf.identity(ref_points, name='knn_cent_pt')

        tensors['knn_inp'] = points_inp
        tensors['knn_ref'] = points_ref
        tensors['knn_cent'] = ref_points

    return graph, tensors


def get_feat_vectors(sess, graph, tensors, guide_points):
    feats = sess.run(tensors['feats'], feed_dict={
        tensors['pc_orig']: guide_points,
        tensors['l0_mask']: np.zeros(guide_points.shape[0])})

    return feats[0]


def filter_inputfiles(inputfiles):
    filtered_inputs = []
    for fn in inputfiles:
        success = True
        try:
            data = meshio.loadmesh(fn)
        except:
            success = False
        if success:
            filtered_inputs.append(fn)
    return filtered_inputs


def random_select(iterable):
    length = len(iterable)
    i = rd.randint(0, length - 1)
    return iterable[i]


def create_logger(logfile):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'

    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        filename=logfile)

    return logging.getLogger(__name__)


def attack(inputfiles, model_path, raw_num_points, shape_names, attack_target, **kwargs):
    # Uncomment this line for more logs from Tensorflow
    tf.logging.set_verbosity(tf.logging.ERROR)

    logfile = kwargs.get('logfile')
    logger = create_logger(logfile)

    normalize = kwargs.get('normalize')
    clip_grad = kwargs.get('clip_grad')

    loss_type = kwargs.get('loss_type')
    random_distortions = kwargs.get('random_distortions')
    logits_lower_bound = kwargs.get('logits_lower_bound')
    update_period = kwargs.get('update_period')

    optim_method = kwargs.get('optim_method')
    max_iter = kwargs.get('max_iter')
    learning_rate = kwargs.get('learning_rate')

    cc_knn = kwargs.get('cc_knn')
    cc_chamfer = kwargs.get('cc_chamfer')
    cc_feats = kwargs.get('cc_feats')
    cc_linf = kwargs.get('cc_linf')

    outputdir = kwargs.get('outputdir')

    batch_size = max(random_distortions, 1)
    logger.info('Batch size: {} ({} random distortions)'.format(
        batch_size, 'with' if random_distortions > 0 else 'without'))

    if cc_linf is None:
        cc_linf = 1e6

    logger.info('Number of points selected from point clouds: {}'.format(raw_num_points))

    logger.info('Loading graph at {}...'.format(model_path))
    num_points, normal_channel = detect_model_parameters(model_path)
    graph, tensors = load_graph(model_path, raw_num_points, num_points, random_distortions)

    num_classes = len(shape_names)
    shape2id = {}
    for idx, name in enumerate(shape_names):
        shape2id[name] = idx

    logger.info('Build attack network...')
    with graph.as_default():
        # Input placeholders
        pointcloud_pl = tensors['pc_orig']
        pointnormal_pl = tensors['nv_orig']
        rotate_matrix_pl = tensors['rot_mat']
        l0_mask_pl = tensors['l0_mask']

        # Groundtruth labels
        label_pl = tf.placeholder(tf.int32, shape=(), name='label')
        tensors['label'] = label_pl

        # Attack type
        targeted_attack_pl = tf.placeholder(tf.bool, shape=(), name='targeted_attack')
        tensors['targeted'] = targeted_attack_pl

        # Feature guide
        guide_pl = tf.placeholder(tf.float32, shape=(256,), name='pointcloud_guide_pl')
        clean_pl = tf.placeholder(tf.float32, shape=(256,), name='pointcloud_orig_feats_pl')

        # Logits & probs layer
        logits = tensors['logits']
        probs = tensors['probs']

        # Define losses
        losses = {}

        # Define logits loss
        logger.info('  Logit loss type: {}'.format(
            'Carlini and Wagner' if loss_type == 'cw' else 'Cross Entropy'))

        target_onehot = tf.one_hot(label_pl, depth=num_classes)
        if loss_type == 'cw':
            real = tf.reduce_sum(logits * target_onehot, axis=-1)
            other = tf.reduce_max(logits * (1.0 - target_onehot) - 1e6 * target_onehot, axis=-1)

            logits_loss_untarget = tf.reduce_mean(real - other)
            logits_loss_target = tf.reduce_mean(other - real)

            logits_loss = tf.where(targeted_attack_pl, logits_loss_target, logits_loss_untarget)
            logits_loss = tf.math.maximum(logits_loss, tf.constant(logits_lower_bound, dtype=tf.float32))
        else:
            ce_untarget = tf.argmax(logits - 1e6 * target_onehot, axis=-1)
            ce_untarget = tf.one_hot(ce_untarget, depth=num_classes)

            logits_loss_untarget = tf.losses.softmax_cross_entropy(onehot_labels=ce_untarget, logits=logits)
            logits_loss_target = tf.losses.softmax_cross_entropy(onehot_labels=target_onehot, logits=logits)

            logits_loss = tf.where(targeted_attack_pl, logits_loss_target, logits_loss_untarget)
        losses['logits'] = logits_loss

        feats = tensors['feats']
        feats_target = tf.reduce_mean(tf.abs(feats - guide_pl))
        feats_orig = tf.reduce_mean(tf.abs(feats - clean_pl))
        feats_loss = tf.math.maximum(feats_target - feats_orig, tf.constant(0.0))
        feats_loss = tf.where(targeted_attack_pl, feats_loss, tf.zeros_like(feats_loss))
        losses['feat'] = cc_feats * feats_loss

        # Define loss using Chamfer pseudo distance
        pc_orig = tensors['pc_orig']
        pc_pert = tensors['pc_samp']  # To avoid using a lot of memory, only consider sampled points
        chamfer = chamfer_distance(pc_pert, pc_orig)  # only consider distance of pert -> orig
        chamfer_loss = cc_chamfer * chamfer
        losses['chamfer'] = chamfer_loss

        pc_pert = tensors['knn_samp']
        pc_all = tensors['pc_pert']
        knn_distance = knn_outlier_distance(pc_pert, pc_all, k=5, alpha=1.05)
        knn_loss = cc_knn * knn_distance
        losses['knn'] = knn_loss

        # Total attack loss
        attack_loss = tf.add_n(list(losses.values()))
        losses['total'] = attack_loss

        # Define optimizer
        if optim_method == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif optim_method == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif optim_method == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optim_method == 'graddesc':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optim_method == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate)
        elif optim_method == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(optim_method))

        if clip_grad:
            gradvars = optimizer.compute_gradients(attack_loss, var_list=[tensors['pert']])
            gradvars = gradients_clipping(gradvars, pointnormal_pl)
            train_step = optimizer.apply_gradients(gradvars)
        else:
            train_step = optimizer.minimize(attack_loss, var_list=[tensors['pert']])

        tensors['pert_init'] = tf.variables_initializer([tensors['pert']])
        tensors['optim_init'] = tf.variables_initializer(optimizer.variables())

        tensors['init'] = tf.group([tensors['pert_init'], tensors['optim_init']])

    # Build perturbation clipping network
    graph, tensors = build_perturbation_clipping_network(graph, tensors)

    # Build normal estimation network
    graph, tensors = build_normal_estimate_network(graph, tensors)

    # Create a Tensorflow session
    logger.info('Create Tensorflow session...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    with graph.as_default():
        tensors['global_init'] = tf.global_variables_initializer()

    sess = tf.Session(graph=graph, config=config)
    sess.run(tensors['global_init'])

    for fn in inputfiles:
        # First initialize the variables including perturbation and the state of the optimizer
        sess.run(tensors['global_init'])

        l0_mask = np.ones(raw_num_points, dtype=np.float32)

        # Prepare input data (one point cloud a time)
        # Note that the point clouds returned may have different shapes (num_points)
        # Always load normal channels
        groundtruth, pointcloud, pointnormal = load_pointcloud(fn, shape2id, logger)

        pointcloud, pointnormal = random_sample_and_add_points(
            pointcloud, pointnormal, raw_num_points, logger)

        attack_target_name = str(attack_target).lower()
        if attack_target is None or attack_target.lower() == 'none':
            attack_target = -1
            logger.info('Untargeted attack (attack target: {})'.format(attack_target_name))
        elif attack_target_name in shape_names:
            attack_target = shape2id[attack_target.lower()]
            logger.info('Targeted attack (attack target: {})'.format(attack_target_name))
        else:
            raise ValueError('Attack target cannot be recognized: {}'.format(attack_target))

        if attack_target == -1:
            label = groundtruth
            is_targeted = False
        else:
            label = attack_target
            is_targeted = True

        groundtruth_text = shape_names[groundtruth]
        if attack_target_name == groundtruth_text:
            logger.warning('Attack target is equal to groundtruth, skipped')
            continue

        # Get guide feats
        clean_points_path = './data/modelnet40_normal_resampled/'
        target_shape_name = shape_names[label]
        target_file = os.listdir(os.path.join(clean_points_path, target_shape_name))
        target_file = os.path.join(clean_points_path, target_shape_name, random_select(target_file))
        guide_points = meshio.loadmesh(target_file)[..., :3]
        guide_feats = get_feat_vectors(sess, graph, tensors, guide_points)
        clean_feats = get_feat_vectors(sess, graph, tensors, pointcloud)

        # Optimize
        for it in range(0, max_iter + 1):
            if it % update_period == 0:
                logger.info('File {} > Target {} > Iter {} / {}:'.format(
                    os.path.basename(fn),
                    attack_target_name,
                    it,
                    max_iter))

            rot_matrix = generate_random_rotations(batch_size)

            sess.run(train_step, feed_dict={
                pointcloud_pl: pointcloud,
                pointnormal_pl: pointnormal,
                rotate_matrix_pl: rot_matrix,
                l0_mask_pl: l0_mask,
                label_pl: label,
                guide_pl: guide_feats,
                clean_pl: clean_feats,
                targeted_attack_pl: is_targeted})

            if it % update_period == 0:
                sess.run(tensors['pert_assign'], feed_dict={
                    tensors['nv_orig']: pointnormal,
                    tensors['cc_linf']: cc_linf})

                # show modified new infinity norm
                newpert = sess.run(tensors['pert'])
                newpert = np.max(np.sqrt(np.sum(np.square(newpert), axis=-1)))
                logger.info('  Current infinity norm: {}'.format(newpert))

            if it % update_period == 0:
                loss_value = sess.run(losses, feed_dict={
                    pointcloud_pl: pointcloud,
                    pointnormal_pl: pointnormal,
                    rotate_matrix_pl: rot_matrix,
                    l0_mask_pl: l0_mask,
                    label_pl: label,
                    guide_pl: guide_feats,
                    clean_pl: clean_feats,
                    targeted_attack_pl: is_targeted})
                logger.info('  Loss: {}'.format(loss_value))

            if it % update_period == 0:
                prob_value = sess.run(probs, feed_dict={
                    pointcloud_pl: pointcloud,
                    pointnormal_pl: pointnormal,
                    rotate_matrix_pl: rot_matrix,
                    l0_mask_pl: l0_mask})
                predict_id = np.argmax(prob_value, axis=-1)
                predict_text = [shape_names[predid] for predid in predict_id]
                predict_string = ' '.join(['({}, {})'.format(i, t) for i, t in zip(predict_id, predict_text)])
                logger.info('  Predictions: {}'.format(predict_string))

        if outputdir is not None:
            logger.info('Writing result to directory: {}'.format(outputdir))
            os.makedirs(outputdir, exist_ok=True)

            # Get perturbed point cloud
            pointcloud_update, pointcloud_perturb = sess.run(
                [tensors['pc_pert'], tensors['pert']],
                feed_dict={
                    pointcloud_pl: pointcloud,
                    l0_mask_pl: l0_mask})

            # Get estimated point normals
            pointnormal_update = []
            for i in range(raw_num_points):
                est_normal = sess.run(tensors['nv_pert'], feed_dict={
                    tensors['pc_pert_single']: pointcloud_update[i],
                    tensors['pc_orig']: pointcloud,
                    tensors['nv_orig']: pointnormal})
                pointnormal_update.append(est_normal)
            pointnormal_update = np.array(pointnormal_update)

            pointcloud = pointcloud_update
            pointnormal = pointnormal_update

            # Concatenate
            pointcloud = np.concatenate([pointcloud, pointnormal], axis=-1)
            logger.info('  Output pointcloud shape: {}'.format(pointcloud.shape))

            outfnid = re.split('[._]', os.path.basename(fn))[1]
            outfn = '{}_{}_{}.xyz'.format(groundtruth_text, attack_target_name, outfnid)

            outfn = os.path.join(outputdir, outfn)
            logger.info('  Write file {} to {}'.format(fn, outfn))

            meshio.savemesh(outfn, pointcloud)

    sess.close()
    return


def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description='Adversarial Attack against PointNet++',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--raw-num-points', type=int, default=10000,
                        help='Number of points to select from the input point clouds')
    parser.add_argument('--without-normalize', action='store_true',
                        help='No mormalization will be applied on the input point clouds')

    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adadelta', 'adagrad', 'adam', 'graddesc', 'momentum', 'rmsprop'],
                        help='Optimizer to use (from Tensorflow)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer')

    parser.add_argument('--clip-grad', action='store_true',
                        help='Clip the gradients to prevent moving points inside the object')
    parser.add_argument('--target', type=str, default=None,
                        help='Attack target class for targeted attack')

    parser.add_argument('--loss-type', type=str, default='cw', choices=['cw', 'ce'],
                        help='Loss to use, cw: Carlini and Wagner, ce: cross entropy')
    parser.add_argument('--max-iter', type=int, default=2500,
                        help='Max iterations for optimization')
    parser.add_argument('--random-distortions', type=int, default=0,
                        help='Number of random distortions (rotations) applied when optimizing, 0 to disable')
    parser.add_argument('--logits-lower-bound', type=float, default=(-15.0),
                        help='Lower bound of the attack confidence')
    parser.add_argument('--update-period', type=int, default=10,
                        help='Number of iterations to print information')
    parser.add_argument('--cc-knn', type=float, default=5.0,
                        help='Coefficient for kNN smoothing loss')
    parser.add_argument('--cc-chamfer', type=float, default=3.0,
                        help='Coefficient for Chamfer distance')
    parser.add_argument('--cc-feats', type=float, default=0.0,
                        help='Coefficient for feature vector loss')
    parser.add_argument('--cc-linf', type=float, default=0.1,
                        help='Coefficient for infinity norm')
    parser.add_argument('--outputdir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to log file')
    parser.add_argument('files', type=str, nargs='+',
                        help='Point cloud files to process')
    args = parser.parse_args()

    shape_names = [line.strip() for line in
                   open('./labels/shape_names.txt')]

    model_path = args.model
    raw_num_points = args.raw_num_points
    normalize = (not args.without_normalize)
    attack_target = args.target.lower() if args.target is not None else None
    inputfiles = argsutils.get_input_files(args.files)

    attack(inputfiles,
           model_path,
           raw_num_points,
           shape_names,
           attack_target,
           optim_method=args.optimizer,
           learning_rate=args.learning_rate,
           normalize=normalize,
           clip_grad=args.clip_grad,
           loss_type=args.loss_type,
           max_iter=args.max_iter,
           random_distortions=args.random_distortions,
           logits_lower_bound=args.logits_lower_bound,
           update_period=args.update_period,
           cc_knn=args.cc_knn,
           cc_chamfer=args.cc_chamfer,
           cc_feats=args.cc_feats,
           cc_linf=args.cc_linf,
           outputdir=args.outputdir,
           logfile=args.log)


if __name__ == '__main__':
    main()
