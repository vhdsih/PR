import numpy as np
import tensorflow as tf
from data import normalize_paras_file


if __name__ == '__main__':

    input_file = 'data/data.csv'
    result_file = 'result.csv'

    data = np.loadtxt(open(input_file, 'r'), delimiter=",",
                      skiprows=0).astype('float32')

    print('shape of input data:', data.shape)

    paras = np.load(normalize_paras_file)
    std = paras[0]
    mean = paras[1]
    data = (data - mean) / std
    result = []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            'checkpoints/mlp-for-exp4-37625.meta')
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('x').outputs[0]
        y = graph.get_collection('predict_result')[0]
        for i in range(data.shape[0]):
            ndata = data[i].reshape(-1, 81)
            r = sess.run(y, feed_dict={x: ndata})
            result.append(r[0])
        result = np.array(result, dtype=np.int32)
        print(result[0])
        np.savetxt(result_file, result, fmt="%d", delimiter=None)
