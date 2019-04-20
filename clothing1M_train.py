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

"""A binary to train Clothing1M using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim 

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('init_dir', 'pretrained/resnet_own',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir', 'events_ce/clothing1M_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

import clothing1M

def train():
  """Train Clothing1M for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for Clothing1M.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      indices, images, labels = clothing1M.distorted_inputs()
 
    # Build a Graph that computes the logits predictions from the
    # inference model.
    is_training = tf.placeholder(tf.bool)
    logits = clothing1M.inference_resnet_own(images,training=is_training)

    # Calculate loss.
    loss = clothing1M.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = clothing1M.train(loss, global_step) 

    # Calculate prediction
    # acc_op contains acc and update_op. So it is the cumulative accuracy when sess runs acc_op
    # if you only want to inspect acc of each batch, just sess run acc_op[0]
    acc_op = tf.metrics.accuracy(labels, tf.argmax(logits,axis=1))
    tf.summary.scalar('training accuracy', acc_op[0])

    #### build scalffold for MonitoredTrainingSession to restore the variables you wish
    variables_to_restore = []
    variables_to_restore += [var for var in tf.trainable_variables() if 'dense' not in var.name]
    variables_to_restore += [g for g in tf.global_variables() if 'moving_mean' in g.name or 'moving_variance' in g.name]
    for var in variables_to_restore:
      print(var.name)
    ckpt = tf.train.get_checkpoint_state(FLAGS.init_dir)
    init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
         ckpt.model_checkpoint_path, variables_to_restore)
    def InitAssignFn(scaffold,sess):
       sess.run(init_assign_op, init_feed_dict)

    scaffold = tf.train.Scaffold(saver=tf.train.Saver(), init_fn=InitAssignFn)
    
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(tf.get_collection('losses')[0])  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    max_steps = int(math.ceil(clothing1M.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*FLAGS.num_epochs/FLAGS.batch_size))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        scaffold = scaffold,
        hooks=[tf.train.StopAtStepHook(last_step=max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        save_checkpoint_secs=60,
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)) as mon_sess:
      while not mon_sess.should_stop():
        res = mon_sess.run([train_op,acc_op,global_step],feed_dict={is_training:True})

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
