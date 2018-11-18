import tensorflow as tf

from model import MLP
from data import Data


def train(model, train_data, saver, writer_train, writer_test, summary_op, epoches, batch_size, lr):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        last_test_acc = 0
        for epoch in range(epoches):
            # train
            steps = train_data.get_data_size() // batch_size
            loss = 0
            accuracy = 0
            for step in range(steps):
                x, y = train_data.next_batch(batch_size)
                _, l, acc, s = sess.run([model.optimizer_op, model.loss, model.accuracy, summary_op], feed_dict={
                                        model.x: x, model.y: y})
                writer_train.add_summary(
                    s, global_step=model.global_step.eval())
                loss += l
                accuracy += acc
            print(' epoch:', epoch, ' loss:', loss /
                  steps, ' \n Train Accuracy:', accuracy / steps)

            # validation
            _x, _y = train_data.get_test_data()
            new_acc, s = sess.run([model.accuracy, summary_op], feed_dict={
                model.x: _x, model.y: _y})
            writer_test.add_summary(
                s, global_step=model.global_step.eval())

            print(' Test Accuracy:', new_acc)
            if new_acc > last_test_acc:
                last_test_acc = new_acc
                print(' Save it to models', epoch)
                saver.save(sess, 'models/mlp%d-%f-' %
                           (epoch, new_acc), global_step=epoch)


if __name__ == '__main__':
    epoches = 50000
    batch_size = 256
    n_input = 81
    n_hidden1 = 512
    n_hidden2 = 256
    n_output = 10
    lr = 0.005
    acc_test = tf.get_variable(
        'accuracy_test', dtype=tf.int32, initializer=tf.constant(0))
    data_path = 'data/data.csv'
    label_path = 'data/labels.csv'
    train_data = Data(data_path, label_path)
    test_data = Data(data_path, label_path, False)

    model = MLP(n_input, n_hidden1, n_hidden2, n_output, lr)
    saver = tf.train.Saver()
    writer_train = tf.summary.FileWriter(
        logdir='log/train', graph=tf.get_default_graph())
    writer_test = tf.summary.FileWriter(logdir='log/test')

    with tf.name_scope('summary'):
        tf.summary.scalar('loss', model.loss)
        # tf.summary.scalar('lr', model.lr)
        tf.summary.scalar('accuracy', model.accuracy)
        summary_op = tf.summary.merge_all()

    train(model, train_data, saver, writer_train, writer_test,
          summary_op, epoches, batch_size, lr)
