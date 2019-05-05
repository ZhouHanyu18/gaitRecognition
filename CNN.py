import tensorflow as tf
import numpy as np


class CNN(object):
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 50, 6])
        self.y_ = tf.placeholder(dtype=tf.int32, shape=[None])
        self.global_step = tf.train.create_global_step()
        self.x_holder = tf.expand_dims(input=self.x, axis=-1)

    def weight_variable(self, shape, n):
        initial = tf.truncated_normal(shape, stddev=n, dtype=tf.float32)
        return initial

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return initial

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding="SAME", name=name)

    def losses(self, logits, labels):
        with tf.variable_scope("loss") as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                           name="xentropy_per_example")
            loss = tf.reduce_mean(cross_entropy, name="loss")
            # tf.summary.scalar(scope.name + "/loss", loss)  # 保存损失模型

        return loss

    # loss损失值优化
    def trainning(self, loss, learning_rate):
        with tf.name_scope("oprimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss, global_step=self.global_step)

        return train_op

    # 准确率计算
    def evaluation(self, logits, labels):
        with tf.variable_scope("accuracy") as scope:
            correct = tf.nn.in_top_k(logits, labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
            tf.summary.scalar(scope.name + "/accuracy", accuracy)  # 保存准确率模型
        return accuracy

    def build_net(self):
        # 第一层卷积层
        with tf.variable_scope('conv1') as scope:
            w_conv1 = tf.Variable(self.weight_variable([5, 6, 1, 32], 1.0), name="weights", dtype=tf.float32)
            b_conv1 = tf.Variable(self.bias_variable([32]), name="blases", dtype=tf.float32)
            h_conv1 = tf.nn.relu(self.conv2d(self.x_holder, w_conv1) + b_conv1, name="conv1")

        # 第一层池化层
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = self.max_pool_2x2(h_conv1, "pooling1")
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")

        # 第2层卷积层
        with tf.variable_scope('conv1') as scope:
            w_conv2 = tf.Variable(self.weight_variable([5, 6, 32, 64], 1.0), name="weights", dtype=tf.float32)
            b_conv2 = tf.Variable(self.bias_variable([64]), name="blases", dtype=tf.float32)
            h_conv2 = tf.nn.relu(self.conv2d(norm1, w_conv2) + b_conv2, name="conv1")

        # 第2层池化层
        with tf.variable_scope('pooling1_lrn') as scope:
            pool2 = self.max_pool_2x2(h_conv2, "pooling1")
            norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")

        # 第3层卷积层
        with tf.variable_scope('conv1') as scope:
            w_conv3 = tf.Variable(self.weight_variable([5, 6, 64, 128], 1.0), name="weights", dtype=tf.float32)
            b_conv3 = tf.Variable(self.bias_variable([128]), name="blases", dtype=tf.float32)
            h_conv3 = tf.nn.relu(self.conv2d(norm2, w_conv3) + b_conv3, name="conv1")

        # 第3层池化层
        with tf.variable_scope('pooling1_lrn') as scope:
            pool3 = self.max_pool_2x2(h_conv3, "pooling1")
            norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")

        # 全连接层
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(norm3, shape=[-1, 128*2*6])
            w_fc1 = tf.Variable(self.weight_variable([128*2*6, 128], 0.005), name="weights", dtype=tf.float32)
            b_fc1 = tf.Variable(self.bias_variable([128]), name="blases", dtype=tf.float32)
            h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)

        h_fc2_dropout = tf.nn.dropout(h_fc1, 0.5)  # 随机删除神经网络中的部分神经元，防止过拟合

        # 回归层
        with tf.variable_scope("sofemax_liner") as scope:
            weights = tf.Variable(self.weight_variable([128, 10], 0.005), name="softmax_linear", dtype=tf.float32)
            biases = tf.Variable(self.bias_variable([10]), name="biases", dtype=tf.float32)
            train_logits = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name="softmax_linear")

        self.loss = self.losses(train_logits, self.y_)
        self.train_op = self.trainning(self.loss, 0.0001)
        self.acc = self.evaluation(train_logits, self.y_)
        self.merged_summary = tf.summary.merge_all()