"""Routine for decoding the clothing1M TFRecord file format"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from six.moves import xrange
import tensorflow as tf
import imagenet_preprocessing

IMAGE_SIZE = 224
NUM_CLASSES = 14
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN_CLEAN = 47569
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000000  # for semi-supervised finetuning, we only use 100000 for sample balance
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10526 
NUM_FILES_FOR_TRAIN_CLEAN = 48 
NUM_FILES_FOR_TRAIN = 1000  # for semi-supervised finetuning, we only use 100000 for sample balance
NUM_FILES_FOR_EVAL = 11
TRAIN_FILE_CLEAN = 'train_clean'
TRAIN_FILE = 'train'
using_clean = False
#using_clean = True
#using_clean_only = False
using_clean_only = True

if using_clean:
   NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN += NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN_CLEAN

def distorted_inputs(data_dir, batch_size, num_epochs):
   
   filenames = [os.path.join(data_dir, TRAIN_FILE + '.tfrecords-%.4d'%i)
                for i in xrange(NUM_FILES_FOR_TRAIN)]
   if using_clean:
     filenames += [os.path.join(data_dir, TRAIN_FILE_CLEAN + '.tfrecords-%.4d'%i)   
                for i in xrange(NUM_FILES_FOR_TRAIN_CLEAN)]
   if using_clean_only:
     filenames = [os.path.join(data_dir, TRAIN_FILE_CLEAN + '.tfrecords-%.4d'%i)   
                for i in xrange(NUM_FILES_FOR_TRAIN_CLEAN)]

   for f in filenames:
     if not tf.gfile.Exists(f):
       raise ValueError('Failed to find file: ' + f)

   dataset = tf.data.TFRecordDataset(filenames)
   
   def _parse_function(example_proto):
     features = {
        'index': tf.FixedLenFeature((),tf.int64),
        'image': tf.FixedLenFeature((),tf.string),
        'label': tf.FixedLenFeature((),tf.int64),
        'image_width': tf.FixedLenFeature((),tf.int64),
        'image_height': tf.FixedLenFeature((),tf.int64)
     }
     parsed_features = tf.parse_single_example(example_proto, features)
     
     index = tf.cast(parsed_features['index'], tf.int32)
     image = tf.decode_raw(parsed_features['image'], tf.uint8)
     label = tf.cast(parsed_features['label'], tf.int32)
     width = tf.cast(parsed_features['image_width'], tf.int32)
     height = tf.cast(parsed_features['image_height'], tf.int32)
    
     # remember to make sure all gray images in the dataset have been pre-transformed into colorful images 
     image = tf.reshape(image,[height,width,3])
     
     with tf.name_scope('data_augmentation'):
       image = tf.cast(image, tf.float32)
       ratio = tf.constant(IMAGE_SIZE,dtype=tf.float32)/tf.to_float(tf.minimum(width,height))
       new_width = tf.to_int32(tf.ceil(tf.to_float(width)*ratio))
       new_height = tf.to_int32(tf.ceil(tf.to_float(height)*ratio))
       distorted_image = tf.image.resize_images(image, [new_height,new_width])
       distorted_image = tf.random_crop(distorted_image, [IMAGE_SIZE, IMAGE_SIZE, 3])
       distorted_image = tf.image.random_flip_left_right(distorted_image)
       distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
       distorted_image = tf.image.random_contrast(distorted_image,lower=0.6,upper=1.4)
       distorted_image = tf.image.random_saturation(distorted_image, lower=0.6,upper=1.4)
       distorted_image = distorted_image - tf.reshape(tf.constant([123.68, 116.78, 103.94]),[1,1,3]) 
       #distorted_image = imagenet_preprocessing.preprocess_image(image, None, height, width, IMAGE_SIZE, IMAGE_SIZE, 3, True)   
 
     return index, distorted_image, label

   dataset = dataset.map(_parse_function)
   dataset = dataset.shuffle(buffer_size=10000)
   dataset = dataset.batch(batch_size)
   dataset = dataset.repeat(num_epochs)
   
   iterator = dataset.make_one_shot_iterator()
   
   next_indices, next_images, next_labels = iterator.get_next()

   return next_indices, next_images, next_labels

def inputs(data_dir, batch_size, eval_data, ORI=False):
   if not eval_data:
     filenames = [os.path.join(data_dir, TRAIN_FILE + '.tfrecords-%.4d'%i)
             for i in xrange(NUM_FILES_FOR_TRAIN)]
     if using_clean:
       filenames += [os.path.join(data_dir, TRAIN_FILE_CLEAN + '.tfrecords-%.4d'%i)   
               for i in xrange(NUM_FILES_FOR_TRAIN_CLEAN)]
     if using_clean_only:
       filenames = [os.path.join(data_dir, TRAIN_FILE_CLEAN + '.tfrecords-%.4d'%i)   
               for i in xrange(NUM_FILES_FOR_TRAIN_CLEAN)]
   else:
     filenames = [os.path.join(data_dir, 'test.tfrecords-%.4d'%i)
             for i in xrange(NUM_FILES_FOR_EVAL)]

   for f in filenames:
     if not tf.gfile.Exists(f):
       raise ValueError('Failed to find file: ' + f)

   dataset = tf.data.TFRecordDataset(filenames)

   def _parse_function(example_proto):
     features = {
        'index': tf.FixedLenFeature((),tf.int64),
        'image': tf.FixedLenFeature((),tf.string),
        'label': tf.FixedLenFeature((),tf.int64),
        'image_width': tf.FixedLenFeature((),tf.int64),
        'image_height': tf.FixedLenFeature((),tf.int64)
     }
     parsed_features = tf.parse_single_example(example_proto, features)

     index = tf.cast(parsed_features['index'], tf.int32)
     image = tf.decode_raw(parsed_features['image'], tf.uint8)
     label = tf.cast(parsed_features['label'], tf.int32)
     width = tf.cast(parsed_features['image_width'], tf.int32)
     height = tf.cast(parsed_features['image_height'], tf.int32)

     # remember to make sure all gray images in the dataset have been pre-transformed into colorful images 
     image = tf.reshape(image,[height,width,3])


     with tf.name_scope('data_augmentation'):
       image = tf.cast(image, tf.float32)
       ratio = tf.constant(IMAGE_SIZE,dtype=tf.float32)/tf.to_float(tf.minimum(width,height))
       new_width = tf.to_int32(tf.ceil(tf.to_float(width)*ratio))
       new_height = tf.to_int32(tf.ceil(tf.to_float(height)*ratio))
       distorted_image = tf.image.resize_images(image, [new_height,new_width])
       distorted_image = tf.image.resize_image_with_crop_or_pad(distorted_image, IMAGE_SIZE, IMAGE_SIZE)

       if ORI:
         ori_image = distorted_image

       distorted_image = distorted_image - tf.reshape(tf.constant([123.68, 116.78, 103.94]),[1,1,3])  
       #distorted_image = imagenet_preprocessing.preprocess_image(image, None, IMAGE_SIZE, IMAGE_SIZE, 3, False)  
 
     if ORI:
       return index, distorted_image, label, ori_image

     return index, distorted_image, label

   dataset = dataset.map(_parse_function)
   dataset = dataset.batch(batch_size)

   iterator = dataset.make_one_shot_iterator()

   if ORI:
     next_indices, next_images, next_labels, next_ori_images = iterator.get_next()
     return next_indices, next_images, next_labels, next_ori_images
      
   next_indices, next_images, next_labels = iterator.get_next()
   return next_indices, next_images, next_labels
     
