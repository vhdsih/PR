import tensorflow as tf


class MLP:
    def __init__(self, n_input, n_hidden, n_output, lr):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = lr

        self._build()

    def _data(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input])
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, self.n_output])
        self.w1 = tf.get_variable('weight1', shape=[self.n_input, self.n_hidden],
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
        self.b1 = tf.get_variable('bias1', shape=[self.n_hidden],
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
        self.w2 = tf.get_variable('weight2', shape=[self.n_hidden, self.n_output],
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
        self.b2 = tf.get_variable('bias2', shape=[self.n_output],
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
        self.global_step = tf.get_variable('global_step', dtype=tf.int32, trainable=False, initializer=tf.constant(0))

    def _loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

    def _optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimizer_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def _eval(self):
        pred = tf.nn.softmax(self.logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.arg_max(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def _build(self):
        self._data()
        self.hidden = tf.nn.relu(tf.add(tf.matmul(self.x, self.w1), self.b1))
        self.dropout = tf.nn.dropout(self.hidden, 0.1)
        self.logits = tf.add(tf.matmul(self.hidden, self.w2), self.b2)
        self._loss()
        self._optimizer()
        self._eval()

