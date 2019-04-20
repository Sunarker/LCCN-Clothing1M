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
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim 

OneHotCategorical = tf.contrib.distributions.OneHotCategorical

import numpy as np
import pickle

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('init_dir', 'events_ce/clothing1M_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir', 'events_varC/clothing1M_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('transition', 'T.pkl',
                           """A pkl file to save a prior transition matrix""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

import clothing1M

def np_smoothing_eye(num_classes, delta=0.05):
  approx_eye = np.eye(num_classes)*(1-delta) + np.ones([num_classes,num_classes])*delta/num_classes
  return approx_eye

def init_C():
  with tf.Graph().as_default():
    # tf always return the final batch even it is smaller than the batch_size of samples
    indices, images, labels = clothing1M.inputs(eval_data=False,batch_size=FLAGS.batch_size)
    is_training = tf.placeholder(tf.bool)
    logits = clothing1M.inference_resnet_own(images,training=is_training)
    labels_ = tf.nn.softmax(logits)
    
    variables_to_restore = []
    variable_averages = tf.train.ExponentialMovingAverage(
        clothing1M.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.init_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint files found')
        return 
    
      inds = []
      preds = []
      annotations = []
      n_iter = 0 
      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                    start=True))
      while not coord.should_stop():
        try:
          ind, pred, annotation = sess.run([indices,labels_,labels],feed_dict={is_training:False})
          inds.append(ind)
          preds.append(pred)
          annotations.append(annotation)
          n_iter += 1
          if n_iter % 100 == 0:
            print('Iters: %d'%n_iter)
        except Exception as e:  # pylint: disable=broad-except
          coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)  

  inds = np.concatenate(inds,axis=0) 
  preds = np.concatenate(preds,axis=0)
  annotations = np.concatenate(annotations,axis=0)

  est_C = np.zeros((clothing1M.NUM_CLASSES+1,clothing1M.NUM_CLASSES))
  for i in xrange(annotations.shape[0]):
    label_ = np.argmax(preds[i])
    label = annotations[i]
    est_C[label_][label] += 1

  return inds, preds, annotations, est_C

def train(infer_z, noisy_y, C):
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
    preds = tf.nn.softmax(logits)
   
    # approximate Gibbs sampling
    T = tf.placeholder(tf.float32,shape=[clothing1M.NUM_CLASSES+1,clothing1M.NUM_CLASSES],name='transition')
    unnorm_probs = preds * tf.gather(tf.transpose(T,[1,0]),labels)
    probs = unnorm_probs / tf.reduce_sum(unnorm_probs,axis=1,keepdims=True)
    sampler = OneHotCategorical(probs=probs)
    labels_ = tf.stop_gradient(tf.argmax(sampler.sample(),axis=1))
 
    # Calculate loss.
    loss = clothing1M.loss(logits, labels_)

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
    #variables_to_restore += [var for var in tf.trainable_variables() if ('dense' not in var.name and 'logits_T' not in var.name)]
    variables_to_restore += tf.trainable_variables()
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

      alpha = 1.0
      C_init = C.copy()
      trans_init = (C_init + alpha) / np.sum(C_init + alpha, axis=1, keepdims=True) 

      warming_up_step_1 = 20000
      warming_up_step_2 = 1000000
      step = 0
      freq_step = 50000

      ## warming up transition
      with open(FLAGS.transition) as f:
        data = pickle.load(f)
      trans_warming_1 = np.concatenate([np_smoothing_eye(clothing1M.NUM_CLASSES,delta=0.05),np.ones([1,clothing1M.NUM_CLASSES])*1.0/clothing1M.NUM_CLASSES],axis=0)
      trans_warming_2 = data[0].copy()
      trans = data[0].copy()
 
      exemplars = []
      while not mon_sess.should_stop():       
        alpha = 1.0
        if (step >= warming_up_step_2) and (step%freq_step == 0):
          trans = (C + alpha) / np.sum(C + alpha, axis=1, keepdims=True)

        if step < warming_up_step_1:
          res = mon_sess.run([train_op,acc_op,global_step,indices,labels,labels_],feed_dict={is_training:True, T: trans_warming_1})
        elif step < warming_up_step_2:
          res = mon_sess.run([train_op,acc_op,global_step,indices,labels,labels_],feed_dict={is_training:True, T: trans_warming_2})
        else:
          res = mon_sess.run([train_op,acc_op,global_step,indices,labels,labels_],feed_dict={is_training:True, T: trans})
  
        for i in xrange(res[3].shape[0]):
          ind = res[3][i]
          #print(ind,noisy_y[ind],res[4][i])
          assert noisy_y[ind] == res[4][i] 
          C[infer_z[ind]][res[4][i]] -= 1
          assert C[infer_z[ind]][noisy_y[ind]] >= 0
          infer_z[ind] = res[5][i]
          C[infer_z[ind]][res[4][i]] += 1  
          #print(res[4][i],res[5][i])

        step = res[2]
        if step % 1000 == 0:
          print('Counting matrix\n', C)
          print('Counting matrix\n', C_init)
          print('Transition matrix\n', trans)
          print('Transition matrix\n', trans_init)
        
        if step % 20000 == 0:
          exemplars.append([infer_z.keys(), infer_z.values(), C])
   
      with open('varC_learnt_%s.pkl'%FLAGS.transition[:-4],'w') as w:
        pickle.dump(exemplars,w) 

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  if os.path.exists('varC.pkl'):
    with open('varC.pkl') as f:
      inds, preds, annotations, C = pickle.load(f)
  else:
      inds, preds, annotations, C = init_C()
      with open('varC.pkl','w') as w:
        pickle.dump([inds, preds, annotations, C],w)

  #print('indices \n', inds)
  #print('predictions \n', np.argmax(preds,axis=1))
  #print('annotations \n', annotations)
  #print('estimated Counting Matrix \n', C)

  infer_z = dict()
  noisy_y = dict()
  for e in xrange(len(inds)):
    infer_z [inds[e]] = np.argmax(preds[e])
    noisy_y[inds[e]] = annotations[e]

  #for key, value in infer_z.items():
  #  print(key, value)

  train(infer_z,noisy_y, C)

if __name__ == '__main__':
  tf.app.run()
