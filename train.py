import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

DATASET = './dataset'
BATCH_SIZE = 10

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

  # Any preprocessing here ...

  # Creates batches by randomly shuffling tensors
  images, labels = tf.train.shuffle_batch([image, label], batch_size=BATCH_SIZE, capacity=30, num_threads=4,
                                          min_after_dequeue=10)

  # return tf.cast(images, tf.float32), tf.reshape(labels, [BATCH_SIZE])
  # return tf.cast(images, tf.float32), tf.reshape(labels, [BATCH_SIZE])
  return tf.cast(images, tf.float32), tf.reshape(labels, [BATCH_SIZE])


def getImage(img):
  print img[0]
  return img

train_images, train_labels = getDataset('train')
# validation_images, validation_labels = getDataset('validation')


# validation_images, validation_labels = getDataset('validation')

# self.conv1 = nn.Conv2d(3, 6, 5)
# self.pool = nn.MaxPool2d(2, 2)
# self.conv2 = nn.Conv2d(6, 16, 5)
# self.fc1 = nn.Linear(16 * 5 * 5, 120)
# self.fc2 = nn.Linear(120, 84)
# self.fc3 = nn.Linear(84, 3)


inputs = tf.placeholder(tf.float32, (None, 32, 32, 3), name="Input")
labels = tf.placeholder(tf.int32, None, name="Labels")
# inputs = tf.placeholder(dtype=tf.float32)
# labels = tf.placeholder(dtype=tf.float32)

# inputs_reshape = tf.reshape(inputs, [-1, 32, 32, 3])

# print inputs_reshape

# input = tf.placeholder(tf.float32, (1, 32, 32, 3))

conv1 = tf.layers.conv2d(inputs, filters=6, kernel_size=5, strides=1, activation=tf.nn.relu)  # 28 x 28 x 6
print conv1

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2) # 14 x 14 x 6
print pool1

conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, strides=1, activation=tf.nn.relu) # 10 x 10 x 16
print conv2

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2) # 8 x 8 x 16

print pool2
conv2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

conv2_flat_dropout = tf.layers.dropout(conv2_flat)

fc1 = tf.layers.dense(conv2_flat_dropout, 120, activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1, 84, activation=tf.nn.relu)
train_logits = tf.layers.dense(fc2, 2)

# train_logits = tf.multiply(train_logits1, 1., name='output')


print train_logits
print train_labels

# cross_entropy = -tf.reduce_sum(tf.cast(train_labels, tf.float32) * tf.log(fc3))
train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits, labels=labels))

# train_loss = tf.losses.mean_squared_error(train_labels, train_logits)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train_op = optimizer.minimize(train_loss)

correct = tf.nn.in_top_k(train_logits, labels, 1)
correct = tf.cast(correct, tf.float16)
accuracy = tf.reduce_mean(correct, name="accuracy")


print '############'
print train_images


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  try:
    # for step in np.arange(MAX_STEP):
    #   if coord.should_stop():
    #     break
    #   _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
    #
    #   if step % 50 == 0:
    #     print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
    #     summary_str = sess.run(summary_op)
    #     train_writer.add_summary(summary_str, step)
    #
    #   if step % 2000 == 0 or (step + 1) == MAX_STEP:
    #     checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
    #     saver.save(sess, checkpoint_path, global_step=step)


    # imgs, lbls = sess.run([train_images, train_labels])

    for step in range(30001):
      if coord.should_stop():
        break

      imgs, lbls = sess.run([train_images, train_labels])

      _, l, acc, cor = sess.run(
        [train_op, train_loss, accuracy, correct],
        feed_dict={
          inputs: imgs,
          labels: lbls
        }
      )

      # print cor
      if step % 50 == 0:

        print lbls
        print 'Step: %d, Loss=%.4f, train accuracy = %.2f%%' % (step, l, acc * 100.0)


      if step % 2000 == 0:
        checkpoint_path = './checkpoints/model.ckpt'
        saver.save(sess, checkpoint_path, global_step=step)

        tf.train.write_graph(sess.graph_def, 'graph', 'model.ph', False)

  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    coord.request_stop()

  coord.join(threads)


