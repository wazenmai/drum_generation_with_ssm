import math
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.slim as slim
import tf_slim as slim
from tensorflow.python.framework import ops

#from utils import *

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def instance_norm(input, name="instance_norm"):
    with tf.compat.v1.variable_scope(name):
        depth = input.get_shape()[3]
        # scale = tf.compat.v1.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        scale = tf.compat.v1.get_variable("scale" ,[depth], initializer=tf.random_normal_initializer(1.0, 0.02))
        offset = tf.compat.v1.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keepdims=True)
        epsilon = 1e-5
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

# self defined fully connected layer network
def dense(input_, output_dim, stddev=0.02, bias_start=0.0, name="dense"):
    with tf.compat.v1.variable_scope(name):
        matrix = tf.compat.v1.get_variable("Matrix", 
                                 shape=[input_.get_shape()[-1], output_dim], 
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(1.0, stddev=0.02, dtype=tf.float32)
                                )
            
        bias = tf.compat.v1.get_variable("bias", 
                               [output_dim],
                               initializer=tf.constant_initializer(bias_start)
                              )
        return tf.matmul(input_, matrix) + bias

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.compat.v1.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.compat.v1.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

if __name__ == '__main__':
    print(tf.__version__)
    conv2d(1, 2)
