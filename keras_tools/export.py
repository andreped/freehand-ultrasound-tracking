import os
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.keras import backend as K
from os.path import join


def freeze_graph_simple(model_dir, output_node_names, output_filename):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
                          """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    output_graph = output_filename

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


    return output_graph_def


def export_current_model(filename, output_nodes):
    K.set_learning_phase(0) # Turn off learning phase

    # Save graph as well
    print('Writing checkpoint to disk..')

    if not os.path.isdir("_Cache"):
        os.makedirs("_Cache")
    saver = tf.train.Saver()
    sess = K.get_session()
    input_graph_name = 'saved_model.pb'
    saver.save(sess, join('_Cache', 'saved_model')) # Save weights
    graph_io.write_graph(sess.graph, "_Cache", input_graph_name, False)
    graph_io.write_graph(sess.graph, "_Cache", 'saved_model.pbtxt', True)

    if type(output_nodes) is list:
        output = ''
        for x in output_nodes:
            output += x + ','
        output = output[:len(output)-1] # Remove last ,
    else:
        output = output_nodes

    print('Running freeze graph script..')
    freeze_graph_simple('_Cache', output, filename)
    print('Done.')
