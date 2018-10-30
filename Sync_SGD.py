from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os
import numpy as np
import pickle

FLAGS = None
log_dir = '/logdir'
REPLICAS_TO_AGGREGATE = 2



# In[3]:


from datetime import datetime

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
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


def main():
  X_train=np.load("X_train.npy")
  y_train=np.load("y_train.npy")
  X_test=np.load("X_test.npy")
  y_test=np.load("y_test.npy")
  X_devel=np.load("X_devel.npy")
  y_devel=np.load("y_devel.npy")
  (X_train,X_test,X_devel)=add_biastomatrix(X_train,X_test,X_devel)
  # Configure
  config=tf.ConfigProto(log_device_placement=False)

  # Server Setup
  cluster = tf.train.ClusterSpec({
        'ps':['10.24.1.221:2228'],
        'worker':['10.24.1.222:2228','10.24.1.223:2228']
        }) #allows this node know about all other nodes
  if FLAGS.job_name == 'ps': #checks if parameter server
    server = tf.train.Server(cluster,
          job_name="ps",
          task_index=FLAGS.task_index,
          config=config)
    server.join()
  else: #it must be a worker server
    is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
    server = tf.train.Server(cluster,
          job_name="worker",
          task_index=FLAGS.task_index,
          config=config)

    # Graph
    worker_device = "/job:%s/task:%d/cpu:0" % (FLAGS.job_name,FLAGS.task_index)
    with tf.device(tf.train.replica_device_setter(ps_tasks=1,
          worker_device=worker_device)):

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

      #global_step = tf.contrib.framework.get_or_create_global_step()


      # create an optimizer then wrap it with SynceReplicasOptimizer
    optimizer = tf.train.GradientDescentOptimizer(.0001)
    optimizer1 = tf.train.SyncReplicasOptimizer(optimizer,replicas_to_aggregate=REPLICAS_TO_AGGREGATE, total_num_replicas=2)

    opt = optimizer1.minimize(loss,global_step=global_step) # averages gradients
      #opt = optimizer1.minimize(REPLICAS_TO_AGGREGATE*loss,
      #                           global_step=global_step) # hackily sums gradient
    m=X_train.shape[0]
    n_epochs = 10001
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))
    checkpoint_path = "/home/kapilpathak/logistic_regressionl2.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = "./my_logreg_model"
    init = tf.global_variables_initializer()

    # Session
    sync_replicas_hook = optimizer1.make_session_run_hook(is_chief)
    stop_hook = tf.train.StopAtStepHook(last_step=10000000)
    hooks = [sync_replicas_hook,stop_hook]

    # Monitored Training Session
    sess = tf.train.MonitoredTrainingSession(master = server.target,
          is_chief=is_chief,
          config=config,
          hooks=hooks,
          stop_grace_period_secs=10)

    print('Starting training on worker %d'%FLAGS.task_index)
    while not sess.should_stop():


        for epoch in range(0, n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train, y_train, batch_size)
                sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
            test_loss,summary_test = sess.run([loss,merged], feed_dict={X: X_test, y: y_test})
            test_writer.add_summary(summary_test, epoch)
            devel_loss,summary_devel=sess.run([loss,merged],feed_dict={X:X_devel,y:y_test})
            devel_writer.add_summary(summary_devel,epoch)
            if epoch % 50 == 0:
                train_loss,summary_train=sess.run([loss,merged],feed_dict={X:X_train,y:y_train})
                train_writer.add_summary(summary_train,epoch)
                print("Epoch:", epoch, "\tTest Loss:", test_loss,"\t Devel Loss:",
                      devel_loss,"\tTrain Loss:",train_loss)

                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))


    if is_chief:
        time.sleep(1)
    print('Done',FLAGS.task_index)

    time.sleep(10) #grace period to wait before closing session
    sess.close()
    print('Session from worker %d closed cleanly'%FLAGS.task_index)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Flags for defining the tf.train.ClusterSpec
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
  print(FLAGS.task_index)

  main()
