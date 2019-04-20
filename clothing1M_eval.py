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

"""Evaluation for Clothing1M.

Accuracy:
clothing1M_train.py achieves 83.0% accuracy after 100K steps (256 epochs
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'events_ce/clothing1M_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'events_ce/clothing1M_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

import clothing1M

def eval_once(saver, summary_writer, top_k_op, summary_op, is_training, logits, labels):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    acc_op: Accuracy op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # add a new summary to save precision
    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    predictions = None
    flag = False

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                  start=True))
    while not coord.should_stop():
      try: 
        prediction,logits_,labels_ = sess.run([top_k_op,logits,labels],feed_dict={is_training:False})
        if flag:
          predictions = np.concatenate((predictions,prediction),axis=0)
        else:
          predictions = prediction
          #print(logits_, labels_, prediction)
          flag = True
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)
    
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    
    precision = np.mean(predictions) 
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    summary.value.add(tag='Precision @ 1', simple_value=precision)
    summary_writer.add_summary(summary, global_step)

def evaluate():
  """Eval Clothing1M for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for Clothing1M.
    eval_data = FLAGS.eval_data == 'test'
    # set eval_batch_size as the number that can be divided by the whole number of eval_data to compat with tf.data.Iterator
    eval_batch_size = FLAGS.batch_size
    indices, images, labels = clothing1M.inputs(eval_data=eval_data,batch_size=eval_batch_size)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    is_training = tf.placeholder(tf.bool)
    logits = clothing1M.inference_resnet_own(images,training=is_training)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits,labels,1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        clothing1M.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    for i in variables_to_restore.keys():
       print(i)

    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op, is_training, logits, labels)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()
