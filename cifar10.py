# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import numpy as np

from six.moves import urllib
import tensorflow as tf
import matplotlib.pyplot as plt

import cifar10_input
from cifar10WithNumpy import main

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
tf.app.flags.DEFINE_string('pm', '33331', 'pooling scheme across scales.  Each number specifies the number of scales remaining at each layer. The first number has to be the same as used in --num_scales.')
tf.app.flags.DEFINE_integer('conv_kernel', 5, 'Size of convolutional kernel')
tf.app.flags.DEFINE_integer('pool_kernel', 3, 'Size of spatial pooling kernel')
tf.app.flags.DEFINE_integer('feats_per_layer', 32, 'Number of feature channels at each layer')
tf.app.flags.DEFINE_boolean('total_pool', False, 'If true, pool all feature maps to 1x1 size in final layer')
tf.app.flags.DEFINE_integer('pool_stride', '1', 'If 2, we get progressive pooling - with overlap pooling, AlexNet style')


# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 10**-4      # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.int32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.int32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  # if not FLAGS.data_dir:
  #   raise ValueError('Please supply a data_dir')
  # data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')

  imagesFromNumpy,labelsFromNumpy,oneHotEncoding=main()
  print(imagesFromNumpy.shape)
  print(labelsFromNumpy.shape)
  # plt.figure()
  # plt.imshow(imagesFromNumpy[0][0])
  # plt.show()
 

  for i in range(2):
    print("------------------------------------------")
    imagePlace=tf.constant(imagesFromNumpy[i*10000:((i+1)*10000)],name="cifar10image",dtype=tf.float32)
    labelPlace=tf.constant(labelsFromNumpy[i*10000:((i+1)*10000)],name="cifar10label",dtype=tf.int32)

  # imagePlace=tf.convert_to_tensor(imagesFromNumpy,name="cifar10image")
  # labelPlace=tf.convert_to_tensor(labelsFromNumpy,name="cifar10label")
  # labelPlace=tf.constant(labelsFromNumpy,shape=(50000),name="cifar10label")
  # imagePlace = tf.placeholder(tf.float32, shape=(50000, 3,32,32,3),name="cifar10image")
  # labelPlace=tf.placeholder(tf.float32, shape=(50000),name="cifar10label")


    images, labels = cifar10_input.distorted_inputs(image=imagePlace,label=labelPlace,
                                                  batch_size=FLAGS.batch_size)
  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      # set up your session here....
    sess.run([images,labels])
    # sess.run(labels, feed_dict={labelPlace: labelsFromNumpy})
    
    coord.request_stop()
    coord.join(threads)
    print("has it terminated")

  
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  # tf.cast(initial,dtype=tf.float64)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  # tf.cast(initial,dtype=tf.float64)
  return tf.Variable(initial)

def conv_scale(x, W):
  return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='VALID')

def inference(x):
  # global out
  print(x.shape)
  if '-' in FLAGS.pm:
    FLAGS.pm= FLAGS.pm.split('-')

  num_layers = len(FLAGS.pm) - 1
  print(x.shape)
  for l in range(num_layers):
    with tf.variable_scope('layer{}'.format(l)):
      with tf.variable_scope('conv'):
        if l == 0:
          bottom = x
          # print(bottom)
          W = weight_variable([1, FLAGS.conv_kernel, FLAGS.conv_kernel, 3, FLAGS.feats_per_layer])
          print(x)
          print("This is kernel")
          print(FLAGS.conv_kernel)
        else:
          if out.get_shape()[2] < FLAGS.conv_kernel:
            bottom = out # l (not l + 1) because from previous layer
            W = weight_variable([1, 1, 1, FLAGS.feats_per_layer, FLAGS.feats_per_layer])
          else:
            bottom = out # l (not l + 1) because from previous layer
            W = weight_variable([1, FLAGS.conv_kernel, FLAGS.conv_kernel, FLAGS.feats_per_layer, FLAGS.feats_per_layer])

        b = bias_variable([FLAGS.feats_per_layer])
        Wx_b = tf.nn.conv3d(bottom, W, strides=[1,1,1,1,1], padding='VALID') + b

        out = tf.nn.relu(Wx_b)
        # norm=tf.nn.lrn(out,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #             name='norm')
        # pool = tf.nn.max_pool(norm, ksize=[1, 3, 3, 1],
        #                  strides=[1, 2, 2, 1], padding='SAME', name='pool')
        shape = out.get_shape()
        print('conv{}'.format(l+1))
        print('\t{} --> {}'.format(bottom.name, out.name))
        print('\t{} --> {}'.format(bottom.get_shape(), out.get_shape()))

      with tf.variable_scope('pool'):
        bottom = out
        if l == num_layers - 1 and FLAGS.total_pool:
          kernel_size = bottom.get_shape()[2]
          out = tf.nn.max_pool3d(bottom, ksize=[1,1, kernel_size, kernel_size,1], strides=[1,1,1,1,1], padding='VALID')
        else:
          out = tf.nn.max_pool3d(bottom, ksize=[1,1, FLAGS.pool_kernel, FLAGS.pool_kernel,1], strides=[1,1,FLAGS.pool_stride,FLAGS.pool_stride,1], padding='VALID')
        shape = out.get_shape()
        print('pool{}'.format(l + 1))
        print('\t{} --> {}'.format(bottom.name, out.name))
        print('\t{} --> {}'.format(bottom.get_shape(), out.get_shape()))
      with tf.variable_scope('scale'):
        bottom = out
        if FLAGS.pm[l + 1]  == FLAGS.pm[l]:
          kernel_size = 1 # useless 1x1 pooling
        elif int(FLAGS.pm[l + 1]) < int(FLAGS.pm[l]):
          print("this is l: "+ str(l))
          num_scales_prev = int(FLAGS.pm[l])
          num_scales_current = int(FLAGS.pm[l + 1])
          print(num_scales_prev)
          print(num_scales_current)
          kernel_size = (num_scales_prev - num_scales_current) + 1
        else:
          raise ValueError('Number of scales must stay constant or decrease, got {}'.format(FLAGS.pm))
        print("this is " + str(kernel_size))
        out = tf.nn.max_pool3d(bottom, ksize=[1,kernel_size,1,1,1], strides=[1,1,1,1,1], padding='VALID')
        shape = out.get_shape()
        print('scale{}'.format(l + 1))
        print('\t{} --> {}'.format(bottom.name, out.name))
        print('\t{} --> {}'.format(bottom.get_shape(), out.get_shape()))

  with tf.variable_scope('fully_connected'):
    bottom = out
    bottom_shape = bottom.get_shape().as_list()
    reshape = tf.reshape(
        bottom,
        [-1, bottom_shape[1] * bottom_shape[2] * bottom_shape[3] * bottom_shape[4]])

    W_fc1 = weight_variable([bottom_shape[1] * bottom_shape[2] * bottom_shape[3] * bottom_shape[4], NUM_CLASSES])
    b_fc1 = bias_variable([NUM_CLASSES])
    out = tf.matmul(reshape, W_fc1) + b_fc1
    print('fc')
    print('\t{} --> {}'.format(bottom.name, out.name))
    print('\t{} --> {}'.format(bottom.get_shape(), out.get_shape()))
    if isinstance(FLAGS.pm, list):
      FLAGS.pm = '-'.join(FLAGS.pm)
    return out

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)