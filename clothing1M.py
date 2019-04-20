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

"""Builds the Clothing1M network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 indices, inputs, labels = distorted_inputs()

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

from six.moves import urllib
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

import clothing1M_input

import resnet_model

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 5,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_string('data_dir', 'data/clothing1M/tfrecords',
                           """Path to the clothing1M data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('resnet_type', 50,
                            """resnet type: 18, 34, 50, 101, 152, 200""") 
# Global constants describing the Clothing1M data set.
IMAGE_SIZE = clothing1M_input.IMAGE_SIZE
NUM_CLASSES = clothing1M_input.NUM_CLASSES
using_clean_only = clothing1M_input.using_clean_only
if using_clean_only:
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = clothing1M_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN_CLEAN
else:
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = clothing1M_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = clothing1M_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5.0       # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.
WEIGHT_DECAY = 0.001
if using_clean_only:
  WEIGHT_DECAY = 0.005

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.

def distorted_inputs():
  """Construct distorted input for Clothing1M training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  indices, images, labels = clothing1M_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                                  batch_size=FLAGS.batch_size,
                                                  num_epochs=FLAGS.num_epochs)

  tf.summary.image('images',images)

  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return indices, images, labels

def inputs(eval_data,batch_size=FLAGS.batch_size, ORI=False):
  """Construct input for Clothing1M evaluation using the Reader ops.

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
  if ORI:
    indices, images, labels, ori_images = clothing1M_input.inputs(data_dir=FLAGS.data_dir,
                                          batch_size=batch_size,
                                          eval_data=eval_data, ORI=ORI)

    tf.summary.image('images',images)

    if FLAGS.use_fp16:
      images = tf.cast(images, tf.float16)
      labels = tf.cast(labels, tf.float16)
    return indices, images, labels, ori_images

  indices, images, labels = clothing1M_input.inputs(data_dir=FLAGS.data_dir,
                                        batch_size=batch_size,
                                        eval_data=eval_data, ORI=ORI)

  tf.summary.image('images',images)

  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return indices, images, labels


def inference_resnet_own(images,training):
  # reference: https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_main.py
  resnet_size = FLAGS.resnet_type
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }
 
  if resnet_size < 50:
     bottleneck = False
     final_size = 512
  else:
     bottleneck = True
     final_size = 2048
 
  resnet = resnet_model.Model(resnet_size=resnet_size,
                        bottleneck = bottleneck,
                        num_classes = NUM_CLASSES+1,
                        num_filters = 64,
                        kernel_size = 7,
                        conv_stride = 2,
                        first_pool_size = 3,
                        first_pool_stride = 2,
                        block_sizes = choices[resnet_size], 
                        block_strides = [1,2,2,2],
                        final_size = final_size,
                        resnet_version=2)
  softmax_linear = resnet(images,training)
  return softmax_linear

def inference_resnet(images,training):
 # reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v2.py
 blocks = [
   nets.resnet_v2.resnet_v2_block('block1', base_depth=64, num_units=2, stride=2),
   nets.resnet_v2.resnet_v2_block('block2', base_depth=128, num_units=2, stride=2),
   nets.resnet_v2.resnet_v2_block('block3', base_depth=256, num_units=2, stride=2),
   nets.resnet_v2.resnet_v2_block('block4', base_depth=512, num_units=2, stride=1)
 ]
 #logits, _ = nets.resnet_v2.resnet_v2(images,blocks,NUM_CLASSES,training,scope='resnet_v2_18') 
 logits, _ = nets.resnet_v2.resnet_v2_50(images,NUM_CLASSES+1,training)
 logits = tf.squeeze(logits,[1,2])
 return logits

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

  l2_loss = tf.add_n([WEIGHT_DECAY * tf.nn.l2_loss(tf.cast(v, tf.float32)) 
                           for v in tf.trainable_variables() if 'batch_normalization' not in v.name],name='l2_loss')

  tf.add_to_collection('losses', cross_entropy_mean)
  tf.add_to_collection('losses', l2_loss)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in Clothing1M model.

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

def train(total_loss, global_step, variable_list=None, return_variable_averages=False, return_lr=False):
  """Train Clothing1M model.

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

  #Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  lr = tf.maximum(lr,1e-5) # this is to balance its effect compared with other learning rates of other optimizers
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients. 
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.MomentumOptimizer(lr,0.9)
    grads = opt.compute_gradients(total_loss)
 
  capped_grads = []
  if variable_list:
     for item in grads:
        if item[1] in variable_list:
           capped_grads.append(item)
        else:
           pass
  else:
     capped_grads = grads

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(capped_grads, global_step=global_step)

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
  if variable_list:
     variables_averages_op = variable_averages.apply(variable_list)
  else:
     variables_averages_op = variable_averages.apply(tf.trainable_variables())
  #variables_averages_op = variable_averages.apply(tf.trainable_variables())

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')
 
  return_group = [train_op] 
  if return_variable_averages:
    return_group.append(variable_averages)
  if return_lr:
    return_group.append(return_lr)

  if len(return_group) == 1:
    return train_op
  else:
    return return_group
