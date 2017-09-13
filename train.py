import tensorflow as tf
import os
# from tensorflow.python.framework import graph_util

DATASET = './dataset'
BATCH_SIZE = 10
MAX_STEP = 100001

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

feature = {'train/image': tf.FixedLenFeature([], tf.string),
           'train/label': tf.FixedLenFeature([], tf.int64)}

def getDataset(type_name):

  tfrecord_path = os.path.join(DATASET, type_name + '.tfrecords')
  filename_queue = tf.train.string_input_producer([tfrecord_path])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(serialized_example, features=feature)

  image = tf.decode_raw(features['train/image'], tf.float32)

  # Cast label data into int32
  label = tf.cast(features['train/label'], tf.int32)
  # Reshape image data into the original shape
  image = tf.reshape(image, [32, 32, 3])

  images, labels = tf.train.shuffle_batch([image, label], batch_size=BATCH_SIZE, capacity=30, num_threads=4,
                                          min_after_dequeue=10)

  return tf.cast(images, tf.float32), tf.reshape(labels, [BATCH_SIZE])


train_images, train_labels = getDataset('train')
validation_images, validation_labels = getDataset('validation')

inputs = tf.placeholder(tf.float32, [None, 32, 32, 3], name="Input")
labels = tf.placeholder(tf.int32, None, name="Labels")

conv1 = tf.layers.conv2d(inputs, filters=6, kernel_size=5, strides=1, activation=tf.nn.relu)  # input 32 x 32 x 3 , output 28 x 28 x 6
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2) # input 28 x 28 x 6, output 14 x 14 x 6
conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, strides=1, activation=tf.nn.relu) # input 14 x 14 x 6 , output 10 x 10 x 16
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2) # input 10 x 10 x 16, output 5 x 5 x 16

conv2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
# conv2_flat_dropout = tf.layers.dropout(conv2_flat)

fc1 = tf.layers.dense(conv2_flat, 120, activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1, 84, activation=tf.nn.relu)
train_logits = tf.layers.dense(fc2, 2)

train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits, labels=labels))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

train_op = optimizer.minimize(train_loss)

tf.summary.scalar('loss', train_loss)

correct = tf.nn.in_top_k(train_logits, labels, 1)
correct = tf.cast(correct, tf.float16)
accuracy = tf.reduce_mean(correct, name="accuracy")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  writer = tf.summary.FileWriter('./log', sess.graph)     # write to file
  merge_op = tf.summary.merge_all() 

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  try:
    for step in range(30001):
      if coord.should_stop():
        break

      imgs, lbls = sess.run([train_images, train_labels])

      _, l = sess.run(
        [train_op, train_loss],
        feed_dict={
          inputs: imgs,
          labels: lbls
        }
      )

      # print cor
      if step % 50 == 0:
        validation_imgs, validation_lbls = sess.run([validation_images, validation_labels])

        acc, result = sess.run(
          [accuracy, merge_op],
          feed_dict={
            inputs: validation_imgs,
            labels: validation_lbls
          }
        )

        writer.add_summary(result, step)
        print 'Step = %d, Loss = %.4f, Accuracy = %.2f%%' % (step, l, acc * 100.0)

      if step % 20000 == 0 or (step + 1) == MAX_STEP:
        checkpoint_path = './checkpoints/model.ckpt'
        saver.save(sess, checkpoint_path, global_step=step)

  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    coord.request_stop()

  coord.join(threads)


