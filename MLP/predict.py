import numpy as np
import tensorflow as tf
from data import normalize_paras_file


model_1 = 'Ms/model_1/mlp420-0.978500--420.meta'
model_1_path = 'Ms/model_1/'
model_2 = 'Ms/model_2/mlp1158-0.984500--1158.meta'
model_2_path = 'Ms/model_2/'
model_3 = 'Ms/model_3/mlp2268-0.984000--2268.meta'
model_3_path = 'Ms/model_3/'


if __name__ == '__main__':
    # path
    input_file = 'data/test.csv'
    result_file = 'result.csv'

    # load the data
    data = np.loadtxt(open(input_file, 'r'), delimiter=",",
                      skiprows=0).astype('float32')
    print('shape of input data:', data.shape)
    # load the paras for normalizing the test data
    paras = np.load(normalize_paras_file)
    std = paras[0]
    mean = paras[1]
    # normalize the input data
    data = (data - mean) / std
    # the results of prediction
    result = []

    with tf.Session() as sess:
        # load model
        saver = tf.train.import_meta_graph(model_2)
        saver.restore(sess, tf.train.latest_checkpoint(model_2_path))
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('x').outputs[0]
        y = graph.get_collection('predict_result')[0]
        # predict
        for i in range(data.shape[0]):
            ndata = data[i].reshape(-1, 81)
            r = sess.run(y, feed_dict={x: ndata})
            result.append(r[0])
        # save it into csv file
        result = np.array(result, dtype=np.int32)
        np.savetxt(result_file, result, fmt="%d", delimiter=None)
