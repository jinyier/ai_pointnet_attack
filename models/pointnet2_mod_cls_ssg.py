#!/usr/bin/env python3
#
# PointNet++ Model for point clouds classification


import tensorflow as tf

from utils import tf_util
from utils.pointnet_util import pointnet_sa_module


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(None, num_point, 3), name='pointclouds')
    labels_pl = tf.placeholder(tf.int32, shape=(None), name='labels')
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1',
                                                       use_nchw=True)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64,
                                                       mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[256, 512, 1024], mlp2=None, group_all=True,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Fully connected layers
    net = tf.layers.flatten(l3_points)
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc0', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp0')
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss, name='classify_loss')
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


def main():
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)


if __name__ == '__main__':
    main()