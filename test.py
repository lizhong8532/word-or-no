import sys
import tensorflow as tf
import cv2
import numpy as np


image_path = sys.argv[1]

# image_data = tf.gfile.FastGFile(image_path, 'rb').read()
img = cv2.imread(image_path)
img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)
# img = img.reshape(img, [1, 32, 32, 3])
img = [img]


with tf.gfile.FastGFile('./graph.pb', 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')


with tf.Session() as sess:
	# for op in sess.graph.get_operations():
	# 	print(op.name, op.values())

	output = sess.graph.get_tensor_by_name('dense_3/BiasAdd:0')
	input = sess.graph.get_tensor_by_name('Input:0')

	predictions = sess.run(output, feed_dict={
		input: img
	})


	# print predictions[0]

	print predictions[0]
	if predictions[0][0] <= 2.5 or predictions[0][1] >= -1:
		print 'word'
	else:
		print 'noWord'