from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""

"""Evaluate eccentricity model
"""

from datetime import datetime
import os.path
import time

import numpy as np
np.random.seed(seed=1)

from six.moves import xrange 
import tensorflow as tf
# import convert_to_records as records
# import ecc
import json 

import cifar10
import cifar10_train


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")

# tf.app.flags.DEFINE_boolean('log_device_placement', False,
#                             """Whether to log device placement.""")
# tf.app.flags.DEFINE_integer('batch_size', 128,
#                             """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('eval_iter', 4180,
                            "Iteration at which to evaluate model")

tf.app.flags.DEFINE_string('eval_name', 'eval_name', 'Directory in which to save evaluation model')
tf.app.flags.DEFINE_string('model_name', 'tffiles', 'Directory in which to save evaluation model')



def eval_dir():
  return os.path.join(FLAGS.train_dir, 'eval', FLAGS.eval_name)

def evaluate():
  """Eval eccentricity model"""

  # dump settings to a JSON
  settings = FLAGS.__dict__
  for key in settings['__flags'].keys():
    if isinstance(settings['__flags'][key], np.floating):
      settings['__flags'][key] = float(settings['__flags'][key])
    elif isinstance(settings['__flags'][key], np.integer):
      settings['__flags'][key] = int(settings['__flags'][key])

  # json.dumps(settings, os.path.join(eval_dir(), 'settings.json'), 
  #   ensure_ascii=True, indent=4, sort_keys=True)  

  print('Settings used are:')
  f = FLAGS.__dict__['__flags']
  for key in sorted(f.keys()):
    print('{} : {}'.format(key, f[key]))
    
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels =  cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Create a saver.
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))

    saver.restore(sess, 
      os.path.join(FLAGS.train_dir, '/tmp/cifar10_train', FLAGS.model_name, 
        'tffiles', 'model.ckpt-{}'.format(FLAGS.eval_iter)))
    
    for var in tf.all_variables():
      try:
        sess.run(var)
      except tf.errors.FailedPreconditionError:
        print('*'*70)
        print(var)

    sess.run(tf.initialize_local_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    summary_writer = tf.train.SummaryWriter(
      os.path.join(eval_dir(), 'tffiles'), sess.graph)

    try:
      step = 0
      true_count = 0
      start_time = time.time()

      while not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        
        step += 1
 
    except tf.errors.OutOfRangeError:
      print('Done evaluating for 1 epoch, %d steps.' % (step))
      # Compute precision @ 1.
      total_sample_count = step * FLAGS.batch_size
      precision = true_count / total_sample_count
      duration = time.time() - start_time

      print('Duration %s: precision @ 1 = %.3f' % (duration, precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, 4180)
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    return precision


def main(argv=None): 
  cifar10.maybe_download_and_extract()

  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  precision=evaluate()
  return precision
  # if tf.gfile.Exists(eval_dir()):
  #   tf.gfile.DeleteRecursively(eval_dir())
  # tf.gfile.MakeDirs(eval_dir())
  # precision = evaluate()
  # return precision 

if __name__ == '__main__':
  tf.app.run()

##Original code for Cifar10 Eval is below

# parser = cifar10.parser

# parser.add_argument('--eval_dir', type=str, default='/tmp/cifar10_eval',
#                     help='Directory where to write event logs.')

# parser.add_argument('--eval_data', type=str, default='test',
#                     help='Either `test` or `train_eval`.')

# parser.add_argument('--checkpoint_dir', type=str, default='/tmp/cifar10_train',
#                     help='Directory where to read model checkpoints.')

# parser.add_argument('--eval_interval_secs', type=int, default=60*5,
#                     help='How often to run the eval.')

# parser.add_argument('--num_examples', type=int, default=10000,
#                     help='Number of examples to run.')

# parser.add_argument('--run_once', type=bool, default=False,
#                     help='Whether to run eval only once.')


# def eval_once(saver, summary_writer, top_k_op, summary_op):
#   """Run Eval once.

#   Args:
#     saver: Saver.
#     summary_writer: Summary writer.
#     top_k_op: Top K op.
#     summary_op: Summary op.
#   """
#   with tf.Session() as sess:
#     ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
#     if ckpt and ckpt.model_checkpoint_path:
#       # Restores from checkpoint
#       saver.restore(sess, ckpt.model_checkpoint_path)
#       # Assuming model_checkpoint_path looks something like:
#       #   /my-favorite-path/cifar10_train/model.ckpt-0,
#       # extract global_step from it.
#       global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#     else:
#       print('No checkpoint file found')
#       return

#     # Start the queue runners.
#     coord = tf.train.Coordinator()
#     try:
#       threads = []
#       for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
#         threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
#                                          start=True))

#       num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
#       true_count = 0  # Counts the number of correct predictions.
#       total_sample_count = num_iter * FLAGS.batch_size
#       step = 0
#       while step < num_iter and not coord.should_stop():
#         predictions = sess.run([top_k_op])
#         true_count += np.sum(predictions)
#         step += 1

#       # Compute precision @ 1.
#       precision = true_count / total_sample_count
#       print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

#       summary = tf.Summary()
#       summary.ParseFromString(sess.run(summary_op))
#       summary.value.add(tag='Precision @ 1', simple_value=precision)
#       summary_writer.add_summary(summary, global_step)
#     except Exception as e:  # pylint: disable=broad-except
#       coord.request_stop(e)

#     coord.request_stop()
#     coord.join(threads, stop_grace_period_secs=10)


# def evaluate():
#   """Eval CIFAR-10 for a number of steps."""
#   with tf.Graph().as_default() as g:
#     # Get images and labels for CIFAR-10.
#     eval_data = FLAGS.eval_data == 'test'
#     images, labels = cifar10.inputs(eval_data=eval_data)

#     # Build a Graph that computes the logits predictions from the
#     # inference model.
#     logits = cifar10.inference(images)

#     # Calculate predictions.
#     top_k_op = tf.nn.in_top_k(logits, labels, 1)

#     # Restore the moving average version of the learned variables for eval.
#     variable_averages = tf.train.ExponentialMovingAverage(
#         cifar10.MOVING_AVERAGE_DECAY)
#     variables_to_restore = variable_averages.variables_to_restore()
#     saver = tf.train.Saver(variables_to_restore)

#     # Build the summary operation based on the TF collection of Summaries.
#     summary_op = tf.summary.merge_all()

#     summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

#     while True:
#       eval_once(saver, summary_writer, top_k_op, summary_op)
#       if FLAGS.run_once:
#         break
#       time.sleep(FLAGS.eval_interval_secs)


# def main(argv=None):  # pylint: disable=unused-argument
#   cifar10.maybe_download_and_extract()
#   if tf.gfile.Exists(FLAGS.eval_dir):
#     tf.gfile.DeleteRecursively(FLAGS.eval_dir)
#   tf.gfile.MakeDirs(FLAGS.eval_dir)
#   evaluate()


# if __name__ == '__main__':
#   FLAGS = parser.parse_args()
#   tf.app.run()
