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
import pickle 
import Image
import os

import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_data', 'train_eval',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'events_ce/clothing1M_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('outlier_dir', 'outlier_images',
                           """Directory where to save images.""")
tf.app.flags.DEFINE_float('threshold', 0.5,
                            """How to define the outlier.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

import clothing1M

def evaluate():
  """Eval Clothing1M for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for Clothing1M.
    eval_data = FLAGS.eval_data == 'test'
    # set eval_batch_size as the number that can be divided by the whole number of eval_data to compat with tf.data.Iterator
    eval_batch_size = FLAGS.batch_size
    indices, images, labels, ori_images = clothing1M.inputs(eval_data=eval_data,batch_size=eval_batch_size, ORI=True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = clothing1M.inference_resnet_own(images,training=False)

    preds = tf.nn.softmax(logits)    

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        clothing1M.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    #for i in variables_to_restore.keys():
    #   print(i)

    saver = tf.train.Saver(variables_to_restore)

    results_list = dict()
    threshold = FLAGS.threshold
    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return
      max_steps = int(math.ceil(clothing1M.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/FLAGS.batch_size))
      for step in xrange(max_steps):
        if step % 100 == 0:
          print('step: %d,'%step, 'number of outliers: %d'%len(results_list))
        res = sess.run([indices, preds, labels, ori_images])
        for ind in xrange(res[0].shape[0]):
          if res[1][ind][-1] > threshold:
            results_list[res[0][ind]] = [res[1][ind], res[2][ind], res[3][ind]]
    
    with open('outlier.pkl','wb') as w:
      pickle.dump(results_list,w)

    for ind in results_list.keys():
      img = Image.fromarray(results_list[ind][-1].astype(np.uint8))
      img.save(os.path.join(FLAGS.outlier_dir, str(ind) + '.jpg'))

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.outlier_dir):
    tf.gfile.DeleteRecursively(FLAGS.outlier_dir)
  tf.gfile.MakeDirs(FLAGS.outlier_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()
