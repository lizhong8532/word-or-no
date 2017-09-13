import tensorflow as tf
import os
import sys
import numpy as np
import cv2
import random

# 0 - noWord
# 1 - word
classes = ['noWord', 'word']
validation_rate = 0.1

dataset_dir = './dataset'

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filename = 'train.tfrecords'
validation_filename = 'validation.tfrecords'


def load_image(addr):
  # read an image and resize to (224, 224)
  # cv2 load images as BGR, convert it to RGB
  img = cv2.imread(addr)
  img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img.astype(np.float32)
  return img


def create(files, name):
  f_name = os.path.join(dataset_dir, name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(f_name)

  for i in range(len(files)):
    label = files[i][1]
    # print how many images are saved every 1000 images
    print 'Name: %s, %d/%d, Label: %d' % (name, i + 1, len(files), label)
    sys.stdout.flush()
    # Load the image
    img = load_image(files[i][0])
    # Create a feature

    feature = {
      'train/label': _int64_feature(label),
      'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))
    }
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

  writer.close()
  sys.stdout.flush()


file_path = './isWord'
files = []

for i in range(len(classes)):
  cl = classes[i]
  dir_path = os.path.join(file_path, cl)
  files_list = os.listdir(dir_path)
  for file_name in files_list:
    _, file_extension = os.path.splitext(file_name)
    if file_extension == '.jpg':
      full_path = os.path.join(dir_path, file_name)
      files.append((full_path, i))


random.seed(0)
random.shuffle(files)

split_num = int(len(files) * validation_rate)

create(files[split_num:], 'train')
create(files[:split_num], 'validation')