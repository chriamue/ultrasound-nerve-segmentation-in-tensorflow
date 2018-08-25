
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
#sys.path.append('../')
from ..model import nerve_net
from ..utils.experiment_manager import make_checkpoint_path

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints',
                            """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 2,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 500000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('keep_prob', 0.7,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate', 1e-5,
                            """ keep probability for dropout """)




def train(run='run', results_dir="results/", batch_size=1, epochs=5, config={'learn_rate':0.001}):
  """Train ring_net for a number of steps."""

  TRAIN_DIR = make_checkpoint_path(results_dir, FLAGS)
  print(TRAIN_DIR)

  with tf.Graph().as_default():
    print('batch size', batch_size)
    # make inputs
    image, mask = nerve_net.inputs(batch_size) 
    # create and unrap network
    prediction = nerve_net.inference(image, FLAGS.keep_prob) 
    # calc error
    error = nerve_net.loss_image(prediction, mask) 
    # train hopefuly 
    train_op = nerve_net.train(error, FLAGS.learning_rate)
    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())

    #for i, variable in enumerate(variables):
    #  print '----------------------------------------------'
    #  print variable.name[:variable.name.index(':')]

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()


    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # load checkpoint
    ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph_def=graph_def)

    for step in range(epochs*5500):
      t = time.time()
      _ , loss_value = sess.run([train_op, error],feed_dict={})
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%100 == 0:
        summary_str = sess.run(summary_op, feed_dict={})
        summary_writer.add_summary(summary_str, step) 
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))

      if step%1000 == 0:
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + TRAIN_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(TRAIN_DIR):
    tf.gfile.DeleteRecursively(TRAIN_DIR)
  tf.gfile.MakeDirs(TRAIN_DIR)
  train()

if __name__ == '__main__':
  tf.app.run()
