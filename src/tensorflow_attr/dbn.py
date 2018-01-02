"""
Deep Belief Network
author: Ye Hu
2016/12/20
"""
import timeit
import numpy as np
import tensorflow as tf
import test.deep_tuto.models.input_data as input_data
from test.deep_tuto.models.logisticRegression import LogisticRegression
from test.deep_tuto.models.mlp import HiddenLayer
from test.deep_tuto.models.rbm import RBM

class DBN(object):
    """
    An implement of deep belief network
    The hidden layers are firstly pretrained by RBM, then DBN is treated as a normal
    MLP by adding a output layer.
    """
    def __init__(self, n_in=784, n_out=10, hidden_layers_sizes=[500, 500]):
        """
        :param n_in: int, the dimension of input
        :param n_out: int, the dimension of output
        :param hidden_layers_sizes: list or tuple, the hidden layer sizes
        """
        # Number of layers
        assert len(hidden_layers_sizes) > 0
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []    # normal sigmoid layer
        self.rbm_layers = []   # RBM layer
        self.params = []       # keep track of params for training

        # Define the input and output
        self.x = tf.placeholder(tf.float32, shape=[None, n_in])
        self.y = tf.placeholder(tf.float32, shape=[None, n_out])

        # Contruct the layers of DBN
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
                input_size = n_in
            else:
                layer_input = self.layers[i-1].output
                input_size = hidden_layers_sizes[i-1]
            # Sigmoid layer
            sigmoid_layer = HiddenLayer(inpt=layer_input, n_in=input_size, n_out=hidden_layers_sizes[i],
                                    activation=tf.nn.sigmoid)
            self.layers.append(sigmoid_layer)
            # Add the parameters for finetuning
            self.params.extend(sigmoid_layer.params)
            # Create the RBM layer
            self.rbm_layers.append(RBM(inpt=layer_input, n_visiable=input_size, n_hidden=hidden_layers_sizes[i],
                                        W=sigmoid_layer.W, hbias=sigmoid_layer.b))
        # We use the LogisticRegression layer as the output layer
        self.output_layer = LogisticRegression(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1],
                                                n_out=n_out)
        self.params.extend(self.output_layer.params)
        # The finetuning cost
        self.cost = self.output_layer.cost(self.y)
        # The accuracy
        self.accuracy = self.output_layer.accuarcy(self.y)
    
    def pretrain(self, sess, train_X, train_y, batch_size=50, pretraining_epochs=100, lr=0.1, k=1,
                    display_step=1):
        """
        Pretrain the layers (just train the RBM layers)
        :param sess: tf.Session
        :param X_train: the input of the train set (You might modidy this function if you do not use the desgined mnist)
        :param batch_size: int
        :param lr: float
        :param k: int, use CD-k
        :param pretraining_epoch: int
        :param display_step: int
        """
        print('Starting pretraining...\n')
        start_time = timeit.default_timer()
        batch_num = int(len(train_X) / batch_size)
        # Pretrain layer by layer
        for i in range(self.n_layers):
            cost = self.rbm_layers[i].get_reconstruction_cost()
            train_ops = self.rbm_layers[i].get_train_ops(learning_rate=lr, k=k, persistent=None)
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for j in range(batch_num):
                    x_batch, _ = train_X[j*batch_size: (j+1)*batch_size], train_y[j*batch_size: (j+1)*batch_size]
                    # 训练
                    sess.run(train_ops, feed_dict={self.x: x_batch})
                    # 计算cost
                    avg_cost += sess.run(cost, feed_dict={self.x: x_batch,}) / batch_num
                # 输出
                if epoch % display_step == 0:
                    print("\tPretraing layer {0} Epoch {1} cost: {2}".format(i, epoch, avg_cost))

        end_time = timeit.default_timer()
        print("\nThe pretraining process ran for {0} minutes".format((end_time - start_time) / 60))
    
    def finetuning(self, sess, train_X, train_y, training_epochs=10, batch_size=100, lr=0.1,
                   display_step=1):
        """
        Finetuing the network
        """
        print("\nStart finetuning...\n")
        start_time = timeit.default_timer()
        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(
            self.cost, var_list=self.params)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            batch_num = int(len(train_X) / batch_size)
            for j in range(batch_num):
                x_batch, y_batch = train_X[j*batch_size: (j+1)*batch_size], train_y[j*batch_size: (j+1)*batch_size]
                # 训练
                sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                # 计算cost
                avg_cost += sess.run(self.cost, feed_dict=
                {self.x: x_batch, self.y: y_batch}) / batch_num
            # 输出
            if epoch % display_step == 0:
                val_acc = sess.run(self.accuracy, feed_dict={self.x: test_X,
                                                       self.y: test_y})
                print("\tEpoch {0} cost: {1}, validation accuacy: {2}".format(epoch, avg_cost, val_acc))

        end_time = timeit.default_timer()
        print("\nThe finetuning process ran for {0} minutes".format((end_time - start_time) / 60))

from src.ayonel.LoadData import *
SEG_PROPORTION = 9/10
if __name__ == '__main__':
    attr_dict, label_dict = load_data()

    for org, repo in [('zendframework', 'zendframework')]:
        input_X = attr_dict[org]
        input_X_dim = len(input_X[0])
        input_y = label_dict[org]
        input_y_dim = len(input_y[0])

        seg_point = int(len(attr_dict[org]) * SEG_PROPORTION)

        train_X = attr_dict[org][:seg_point]
        train_y = label_dict[org][:seg_point]

        test_X = attr_dict[org][seg_point:]
        test_y = label_dict[org][seg_point:]

        dbn = DBN(n_in=input_X_dim, n_out=input_y_dim, hidden_layers_sizes=[100, 100,100])
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        # set random_seed
        tf.set_random_seed(seed=1111)
        dbn.pretrain(sess, train_X=train_X, train_y=train_y)
        dbn.finetuning(sess, train_X=train_X, train_y=train_y)