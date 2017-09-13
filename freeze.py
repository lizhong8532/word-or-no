import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(model_folder):
	# We retrieve our checkpoint fullpath
	checkpoint = tf.train.get_checkpoint_state(model_folder)
	input_checkpoint = checkpoint.model_checkpoint_path

	# We precise the file fullname of our freezed graph
	output_graph = "./graph.pb"

	output_node_names = "accuracy"

	# We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
	clear_devices = True

	# We import the meta graph and retrive a Saver
	saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

	# We retrieve the protobuf graph definition
	graph = tf.get_default_graph()
	input_graph_def = graph.as_graph_def()

	with tf.Session() as sess:
		saver.restore(sess, input_checkpoint)

		# We use a built-in TF helper to export variables to constant
		output_graph_def = graph_util.convert_variables_to_constants(
			sess,
			input_graph_def,
			output_node_names.split(",")  # We split on comma for convenience
		)

		# Finally we serialize and dump the output graph to the filesystem
		with tf.gfile.GFile(output_graph, "wb") as f:
			f.write(output_graph_def.SerializeToString())
		print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
	freeze_graph('./checkpoints')