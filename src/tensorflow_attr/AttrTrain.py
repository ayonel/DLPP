'''
 Author: ayonel
 Date: 
 Blog: https://ayonel.me
 GitHub: https://github.com/ayonel
 E-mail: ayonel@qq.com
'''
from src.ayonel.LoadData import *
from src.tensorflow_attr.dbn import DBN
import tensorflow as tf
import test.deep_tuto.models.input_data as input_data




if __name__ == '__main__':
    attr_dict, label_dict = load_data()
    for org, repo in [('zendframework', 'zendframework')]:
        # mnist examples
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        print(mnist.train.num_examples)

        dbn = DBN(n_in=784, n_out=10, hidden_layers_sizes=[10, 10, 10])
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        # set random_seed
        tf.set_random_seed(seed=1111)
        dbn.pretrain(sess, X_train=mnist)
        dbn.finetuning(sess, trainSet=mnist)