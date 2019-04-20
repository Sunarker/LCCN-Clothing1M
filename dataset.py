import tensorflow as tf
import os
from PIL import Image
import random

def RecordData(path,referencePath,save_name,start_count=0):
   # image index and path (image_index:image_path)
   data = dict()
   count = start_count
   with open(path) as f:
     for line in f:
        ipath = line.strip()
        data[count] = ipath
        count += 1

   # image path and label (image_path:image_label)
   referenceData = dict()
   with open(referencePath) as f:
     for line in f:
         ipath, label = line.strip().split(' ')
         referenceData[ipath] = int(label)

   # save the image sample into the TFRecord file (index,image,label,width,height)
   # each TFRecord file has at most 1000 elements and the sample number shold below 1000x9999
   fid = 0
   writer = tf.python_io.TFRecordWriter(save_name + '.tfrecords-%.4d'%fid)
   shuffle_list = range(count-start_count)
   random.shuffle(shuffle_list) 
   for i in xrange(count-start_count):
     if i%1000 == 0 and i > 0:
       writer.close()
       fid = i/1000
       writer = tf.python_io.TFRecordWriter(save_name + '.tfrecords-%.4d'%fid)

     index = shuffle_list[i] + start_count
     ipath = data[index]
     label = referenceData[ipath]
     img = Image.open(ipath,'r')
     if img.mode != 'RGB':
       print img.mode,'to RGB'
       img = img.convert('RGB')
     if i%1000 == 0 or min(img.size[0],img.size[1]) < 256:
       print i,img.size
       pass
     example = tf.train.Example(
        features = tf.train.Features(feature = {
        'index': tf.train.Feature(int64_list = tf.train.Int64List(value=[index])),
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value=[img.tobytes()])),
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
        'image_width': tf.train.Feature(int64_list = tf.train.Int64List(value=[img.size[0]])),
        'image_height': tf.train.Feature(int64_list = tf.train.Int64List(value=[img.size[1]]))
        }))
     writer.write(example.SerializeToString())
   
   writer.close()

#RecordData('noisy_train_key_list.txt','noisy_label_kv.txt','data/clothing1M/tfrecords/train')
RecordData('clean_train_key_list.txt','clean_label_kv.txt','data/clothing1M/tfrecords/train_clean',1000000)
#RecordData('clean_test_key_list.txt','clean_label_kv.txt','data/clothing1M/tfrecords/test')
