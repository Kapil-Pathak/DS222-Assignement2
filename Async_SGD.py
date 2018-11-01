from __future__ import print_function
import argparse
import sys

import tensorflow as tf
import time
import os
import numpy as np
import pickle
FLAGS = None

from datetime import datetime

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs_Async"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


# In[4]:


def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


# In[5]:


def add_biastomatrix(*args):
    matrices=[]
    for X in args:
        m=X.shape[0]
        X=np.c_[np.ones((m,1)),X]
        matrices.append(X)
    return tuple(matrices)


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  X_train=np.load("X_train.npy")
  y_train=np.load("y_train.npy")
  X_test=np.load("X_test.npy")
  y_test=np.load("y_test.npy")
  X_devel=np.load("X_devel.npy")
  y_devel=np.load("y_devel.npy")
  (X_train,X_test,X_devel)=add_biastomatrix(X_train,X_test,X_devel)
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      beta=0.001
      n_inputs = 100
      logdir = log_dir("logreg")
      n_classes=y_train.shape[1]
      X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
      y = tf.placeholder(tf.int32, shape=(None, n_classes), name="y")
      n_inputs_including_bias = int(X.shape[1])

      initializer = tf.random_uniform([n_inputs_including_bias, n_classes], -1.0, 1.0, seed=42)
      theta = tf.Variable(initializer, name="theta")
      logits = tf.matmul(X, theta, name="logits")
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)+beta*tf.nn.l2_loss(theta))
      global_step = tf.contrib.framework.get_or_create_global_step()
      train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
      loss_summary = tf.summary.scalar('Cross-Entropy_Loss', loss)
      #init = tf.global_variables_initializer()
      saver = tf.train.Saver()
      train_writer = tf.summary.FileWriter(logdir+"/train", tf.get_default_graph())
      test_writer=tf.summary.FileWriter(logdir+"/test")
      devel_writer=tf.summary.FileWriter(logdir+"/devel")
      merged=tf.summary.merge_all()
    m=X_train.shape[0]
    n_epochs = 10001
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))
    init = tf.global_variables_initializer()
    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/home/kapilpathak/train_logs_Async",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():

        start_epoch = 0


        for epoch in range(start_epoch, n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train, y_train, batch_size)
                mon_sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
            if epoch % 50 == 0:
                train_loss,summary_train=mon_sess.run([loss,merged],feed_dict={X:X_train,y:y_train})
                train_writer.add_summary(summary_train,epoch)
                print("Epoch:", epoch,"\tTrain Loss:",train_loss)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
