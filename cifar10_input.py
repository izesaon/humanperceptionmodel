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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import scipy as sp
from scipy.misc import toimage
import matplotlib.pyplot as plt
import random

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  # tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  print("HI")
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  # NOTE: since per_image_standardization zeros the mean and makes
  # the stddev unit, this likely has no effect see tensorflow#1458.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # float_image=tf.reshape(float_image,[IMAGE_SIZE*1,IMAGE_SIZE,3])
  # float_image=tf.reshape(float_image,[-1,IMAGE_SIZE,IMAGE_SIZE,3])

  #without padding
  first_crop=crop_center(distorted_image,32,32)
  second_crop=crop_center(distorted_image,28,28)
  third_crop=crop_center(distorted_image,24,24)
  fourth_crop=crop_center(distorted_image,20,20)

  # with padding
  first_crop_with_padding=crop_center(distorted_image,20,20)
  first_crop_with_padding=tf.image.per_image_standardization(first_crop_with_padding)

  read_input.label.set_shape([1])
  label=read_input.label

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      # set up your session here....
    label_class=sess.run(label)
   
    if(label_class==0):
        first_crop_numpy=sess.run(first_crop_with_padding)
        pad = 6 #pixels
        first_crop_numpy = np.pad(first_crop_numpy, ((pad,pad),(pad,pad),(0,0)), 'constant')
        first_crop_numpy=tf.convert_to_tensor(first_crop_numpy,dtype=tf.float32)
        
        first_crop=crop_center(first_crop_numpy,32,32)
        first_crop=sess.run(first_crop)
        first_crop=sp.misc.imresize(first_crop,(IMAGE_SIZE,IMAGE_SIZE))
        first_crop=tf.convert_to_tensor(first_crop,dtype=tf.float32)
        first_crop=tf.image.per_image_standardization(first_crop)

        second_crop=crop_center(first_crop_numpy,28,28)
        second_crop=sess.run(second_crop)
        second_crop=sp.misc.imresize(second_crop,(IMAGE_SIZE,IMAGE_SIZE))
        second_crop=tf.convert_to_tensor(second_crop,dtype=tf.float32)
        second_crop=tf.image.per_image_standardization(second_crop)

        third_crop=crop_center(first_crop_numpy,24,24)
        third_crop=sess.run(third_crop)
        third_crop=sp.misc.imresize(third_crop,(IMAGE_SIZE,IMAGE_SIZE))
        third_crop=tf.convert_to_tensor(third_crop,dtype=tf.float32)
        third_crop=tf.image.per_image_standardization(third_crop)

        fourth_crop=crop_center(first_crop_numpy,20,20)
        fourth_crop=sess.run(fourth_crop)
        fourth_crop=sp.misc.imresize(fourth_crop,(IMAGE_SIZE,IMAGE_SIZE))
        fourth_crop=tf.convert_to_tensor(fourth_crop,dtype=tf.float32)
        fourth_crop=tf.image.per_image_standardization(fourth_crop)

        final_output=tf.stack([first_crop, second_crop, third_crop, fourth_crop])
    

        final_output=tf.reshape(final_output,[IMAGE_SIZE*4,IMAGE_SIZE,3])
        # plt.figure()
        # final_output=sess.run(final_output)
        # plt.imshow(final_output.astype(np.uint8))
        # plt.show()
    
    else:
      print("Label Others")
      first_crop_numpy=sess.run(first_crop)
      first_crop_numpy=sp.misc.imresize(first_crop_numpy,(IMAGE_SIZE,IMAGE_SIZE))
      first_crop_numpy=tf.convert_to_tensor(first_crop_numpy,dtype=tf.float32)
      # first_crop_numpy=tf.image.per_image_standardization(first_crop_numpy)

      second_crop_numpy=sess.run(second_crop)
      second_crop_numpy=sp.misc.imresize(second_crop_numpy,(IMAGE_SIZE,IMAGE_SIZE))
      second_crop_numpy=tf.convert_to_tensor(second_crop_numpy,dtype=tf.float32)
      # second_crop_numpy=tf.image.per_image_standardization(second_crop_numpy)
        
      third_crop_numpy=sess.run(third_crop)
      third_crop_numpy=sp.misc.imresize(third_crop_numpy,(IMAGE_SIZE,IMAGE_SIZE))
      third_crop_numpy=tf.convert_to_tensor(third_crop_numpy,dtype=tf.float32)
      # third_crop_numpy=tf.image.per_image_standardization(third_crop_numpy)

      fourth_crop_numpy=sess.run(fourth_crop)
      fourth_crop_numpy=sp.misc.imresize(fourth_crop_numpy,(IMAGE_SIZE,IMAGE_SIZE))
      fourth_crop_numpy=tf.convert_to_tensor(fourth_crop_numpy,dtype=tf.float32)
      # fourth_crop_numpy=tf.image.per_image_standardization(fourth_crop_numpy)

      final_output=tf.stack([first_crop_numpy, second_crop_numpy, third_crop_numpy, fourth_crop_numpy])
      final_output=tf.reshape(final_output,[IMAGE_SIZE*4,IMAGE_SIZE,3])
      plt.figure()
      final_output=sess.run(final_output)
      plt.imshow(final_output.astype(np.uint8))
      plt.show()

    
    final_output=tf.reshape(final_output,[-1,IMAGE_SIZE,IMAGE_SIZE,3])

    # final_output_numpy=sess.run(final_output)
    # print(final_output.shape)
    # plt.figure()
    # plt.imshow(final_output_numpy.astype(np.uint8))
    # plt.show()

  # Set the shapes of tensors.

  # float_image.set_shape([height, width, 3])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(final_output, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  # Subtract off the mean and divide by the variance of the pixels.

  float_image = tf.image.per_image_standardization(resized_image)

  first_crop=crop_center(resized_image,32,32)
  second_crop=crop_center(resized_image,32,32)
  third_crop=crop_center(resized_image,32,32)
  fourth_crop=crop_center(resized_image,32,32)


  with tf.Session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      # set up your session here....
      first_crop_numpy=sess.run(first_crop)
      first_crop_numpy=sp.misc.imresize(first_crop_numpy,(IMAGE_SIZE,IMAGE_SIZE))
      first_crop_numpy=tf.convert_to_tensor(first_crop_numpy,dtype=tf.float32)
      # first_crop_numpy=tf.image.per_image_standardization(first_crop_numpy)

      second_crop_numpy=sess.run(second_crop)
      second_crop_numpy=sp.misc.imresize(second_crop_numpy,(IMAGE_SIZE,IMAGE_SIZE))
      second_crop_numpy=tf.convert_to_tensor(second_crop_numpy,dtype=tf.float32)
      # second_crop_numpy=tf.image.per_image_standardization(second_crop_numpy)
        
      third_crop_numpy=sess.run(third_crop)
      third_crop_numpy=sp.misc.imresize(third_crop_numpy,(IMAGE_SIZE,IMAGE_SIZE))
      third_crop_numpy=tf.convert_to_tensor(third_crop_numpy,dtype=tf.float32)
      # third_crop_numpy=tf.image.per_image_standardization(third_crop_numpy)

      fourth_crop_numpy=sess.run(fourth_crop)
      fourth_crop_numpy=sp.misc.imresize(fourth_crop_numpy,(IMAGE_SIZE,IMAGE_SIZE))
      fourth_crop_numpy=tf.convert_to_tensor(fourth_crop_numpy,dtype=tf.float32)
      # fourth_crop_numpy=tf.image.per_image_standardization(fourth_crop_numpy)

      final_output=tf.stack([first_crop_numpy, second_crop_numpy, third_crop_numpy, fourth_crop_numpy])
      final_output=tf.reshape(final_output,[IMAGE_SIZE*4,IMAGE_SIZE,3])
      final_output_numpy=sess.run(final_output)
      print(final_output.shape)
      plt.figure()
      plt.imshow(final_output_numpy.astype(np.uint8))
      plt.show()

    
      final_output=tf.reshape(final_output,[-1,IMAGE_SIZE,IMAGE_SIZE,3])


  read_input.label.set_shape([1])
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(final_output, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
