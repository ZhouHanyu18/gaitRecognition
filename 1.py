import os
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import time
from scipy import signal
from PCA_KNN import PCA_KNN
from SVM import SVM
from CNN import CNN

FLAGS = tf.flags.FLAGS

tf.app.flags.DEFINE_integer("layer_num", 1, "number of layer")
tf.app.flags.DEFINE_integer("units_num", 128, "number of hidden units")
tf.app.flags.DEFINE_integer("epoch", 50, "epoch of training step")
tf.app.flags.DEFINE_integer("batch_size", 128, "mini_batch_size")
tf.app.flags.DEFINE_integer("W", 6, "use ten point to predict the value of 11th")
tf.app.flags.DEFINE_integer("H", 50, "use ten point to predict the value of 11th")
tf.app.flags.DEFINE_enum("model_state", "predict", ["train", "predict"], "model state")
tf.app.flags.DEFINE_float("lr", 0.01, "learning rate")


class RNN(object):
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.H, FLAGS.W])
        self.y_ = tf.placeholder(dtype=tf.int32, shape=[None])
        self.global_step = tf.train.create_global_step()
        self.input = self.x

    def build_rnn(self):
        with tf.variable_scope("gru_layer"):
            cells = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(FLAGS.units_num) for _ in range(FLAGS.layer_num)])

            outputs, final_states = tf.nn.dynamic_rnn(cell=cells, inputs=self.input, dtype=np.float32)
            self.outputs = outputs[:, -1]

        with tf.variable_scope("output_layer"):
            self.pre = tf.contrib.layers.fully_connected(self.outputs, 10, activation_fn=None)

    def build_train_op(self):
        with tf.variable_scope("train_op_layer"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pre, labels=self.y_)
            self.loss = tf.reduce_mean(cross_entropy)
            # tf.summary.scalar(name="loss", tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
            self.train_op = optimizer.minimize(self.loss, self.global_step)

    def evaluation(self):
        with tf.variable_scope("accuracy") as scope:
            correct = tf.nn.in_top_k(self.pre, self.y_, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            # tf.summary.scalar(name="accuracy", tensor=accuracy)
            self.acc = accuracy

    def build_net(self):
        self.build_rnn()
        self.build_train_op()
        self.evaluation()
        # self.merged_summary = tf.summary.merge_all()


def get_batches(X, y):
    batch_size = FLAGS.batch_size
    for i in range(0, len(X), batch_size):
        begin_i = i
        end_i = i + batch_size if (i+batch_size) < len(X) else len(X)
        yield X[begin_i:end_i], y[begin_i:end_i]


def get_file():
    file_dir = "data"
    file_dir2 = "data2"
    X = []
    Y = []
    acc = os.listdir(file_dir)
    gyr = os.listdir(file_dir2)
    for i in range(10):
        f = open(file_dir + '/' + acc[i])
        f2 = open(file_dir2 + '/' + gyr[i])
        line = f.readlines()
        line2 = f2.readlines()
        temp = []
        for num in range(len(line)):
            if num < 50:
                continue
            time, x, y, z = [float(i) for i in line[num].split()]
            time2, x2, y2, z2 = [float(i) for i in line2[num].split()]
            temp.append([x, y, z, x2, y2, z2])
            # temp.append([x, y, z])
            num += 1
            if len(temp) == 50:
                X.append(temp)
                Y.append(i)
                temp = temp[25:]
        # b, a = signal.butter(8, 0.02, 'lowpass')
        # temp = signal.filtfilt(b, a, temp, axis=0)
        # group = []
        # for x in temp:
        #     group.append(x)
        #     if len(group) == 50:
        #         X.append(group)
        #         Y.append(i)
        #         group = group[10:]
    return X, Y


if __name__ == "__main__":
    log_dir = "log/"
    X, Y = get_file()
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    print(X.shape)
    print(Y.shape)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=40)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=40)
    print(train_x.shape)
    print(test_x.shape)
    print(valid_x.shape)
    # print("----------------Enter PCA_KNN model----------------")
    # PCA_KNN(train_x, test_x, train_y, test_y)
    # print("----------------Enter SVM model----------------")
    # SVM(train_x, test_x, train_y, test_y)

    # rnn_model = CNN()

    rnn_model = RNN()
    rnn_model.build_net()

    saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=log_dir, is_chief=True, saver=saver, summary_op=None,
                             save_summaries_secs=None,save_model_secs=None, global_step=rnn_model.global_step)
    sess_context_manager = sv.prepare_or_wait_for_session()

    maxAcc = 0
    with sess_context_manager as sess:
        if FLAGS.model_state == "train":
            print("----------------Enter train model----------------")
            print(time.strftime('%Y-%m-%d %H:%M:%S'))
            # summary_writer = tf.summary.FileWriter(log_dir)
            for e in range(FLAGS.epoch):
                train_x, train_y = shuffle(train_x, train_y)
                for xs, ys in get_batches(train_x, train_y):
                    feed_dict = {rnn_model.x: xs, rnn_model.y_: ys}
                    _, loss, step, train_acc = sess.run(
                        [rnn_model.train_op, rnn_model.loss, rnn_model.global_step, rnn_model.acc], feed_dict=feed_dict)
                    if step % 10 == 0:
                        feed_dict = {rnn_model.x: valid_x, rnn_model.y_: valid_y}
                        valid_acc = sess.run(rnn_model.acc, feed_dict=feed_dict)
                        print("epoch->{:<3} step->{:<5} loss:{:<10.5} train_acc:{:<10.2%} "
                              "valid_acc:{:<10.2%} maxAcc:{:<10.2%}".
                              format(e, step, loss, train_acc, valid_acc, maxAcc))
                        # summary_writer.add_summary(merged_summary, step)
                        if valid_acc > maxAcc:
                            maxAcc = valid_acc
                            saver.save(sess=sess, save_path=log_dir, global_step=step)
                            print("●_●")
            print(time.strftime('%Y-%m-%d %H:%M:%S'))
        print("-------------------Enter predict model---------------")
        model_file = tf.train.latest_checkpoint(log_dir)
        saver.restore(sess, model_file)
        feed_dict = {rnn_model.x: test_x, rnn_model.y_: test_y}
        acc = sess.run(rnn_model.acc, feed_dict=feed_dict)
        print("test_acc:{:.2%}".format(acc))
