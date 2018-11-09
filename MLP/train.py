import tensorflow as tf

from model import MLP
from data import Data



def train(model, train_data, test_data, saver, writer, summary_op, epoches, batch_size, save_step):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epoches):
            steps = train_data.get_data_size() // batch_size
            loss = 0
            accuracy = 0
            for step in range(steps):
                x, y = train_data.next_batch(batch_size)
                _, l, acc, s = sess.run([model.optimizer_op, model.loss, model.accuracy, summary_op], feed_dict={model.x: x, model.y: y})
                writer.add_summary(s, global_step=model.global_step.eval())
                loss += l
                accuracy += acc
            print(' epoch:', epoch, ' loss:', loss / steps, ' accuracy:', accuracy / steps)
            
            
            accuracy_test = 0
            steps_test = test_data.get_data_size() // batch_size
            for step in range(steps_test):
                x, y = test_data.next_batch(batch_size)
                acc = sess.run(model.accuracy, feed_dict={model.x: x, model.y: y})
                accuracy_test += acc
            print(' Test:', accuracy_test / steps_test)

            if epoch != 0 and epoch % save_step == 0:
                saver.save(sess, 'checkpoints/mlp-for-exp4', global_step=model.global_step.eval())





if __name__ == '__main__':
    epoches = 500000
    batch_size = 128
    n_input = 81
    n_hidden = 128
    n_output = 10
    lr = 0.01

    save_step = 100
    data_path = 'data/data.csv'
    label_path = 'data/labels.csv'
    train_data = Data(data_path, label_path)
    test_data = Data(data_path, label_path, False)

    model = MLP(n_input, n_hidden, n_output, lr)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logdir='log/', graph=tf.get_default_graph())

    with tf.name_scope('summary'):
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('accuracy', model.accuracy)
        summary_op = tf.summary.merge_all()

    train(model, train_data, test_data, saver, writer, summary_op, epoches, batch_size, save_step)
