import tensorflow as tf


class MLP:
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, lr=0.01):
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.lr = tf.get_variable(
            'lr', initializer=tf.constant(lr, dtype=tf.float32))

        self._build()

    def _data(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[
                                None, self.n_input], name='x')
        self.y = tf.placeholder(dtype=tf.int32, shape=[
                                None, self.n_output], name='y')

        self.w1 = tf.get_variable('weight1', shape=[self.n_input, self.n_hidden1],
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
        self.b1 = tf.get_variable('bias1', shape=[self.n_hidden1],
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

        self.w2 = tf.get_variable('weight2', shape=[self.n_hidden1, self.n_hidden2],
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
        self.b2 = tf.get_variable('bias2', shape=[self.n_hidden2],
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

        self.w3 = tf.get_variable('weight3', shape=[self.n_hidden2, self.n_output],
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
        self.b3 = tf.get_variable('bias3', shape=[self.n_output],
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

        self.global_step = tf.get_variable(
            'global_step', dtype=tf.int32, trainable=False, initializer=tf.constant(0))

    def _loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.y))

    def _optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimizer_op = self.optimizer.minimize(
            self.loss, global_step=self.global_step)

    def _eval(self):
        pred = tf.nn.softmax(self.logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def _predict(self):
        pred = tf.nn.softmax(self.logits)
        predict_result = tf.argmax(pred, 1)
        tf.add_to_collection('predict_result', predict_result)

    def _build(self):
        self._data()

        hidden1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.w1), self.b1))
        dropout1 = tf.nn.dropout(hidden1, 0.90)

        hidden2 = tf.nn.relu(tf.add(tf.matmul(dropout1, self.w2), self.b2))
        dropout2 = tf.nn.dropout(hidden2, 1.0)

        self.logits = tf.add(tf.matmul(dropout2, self.w3), self.b3)

        self._loss()
        self._optimizer()
        self._eval()
        self._predict()
