#!./tensorflow/venv/bin/python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## Parameters ##
LEARN_RATE = 0.065
PATTERN = 'pattern6'
HIDDEN_UNIT = 5
EPOCH = 100
################


def load_dataset():
    with open('patterns/{}.pat'.format(PATTERN)) as f:
        lines = f.read().splitlines()
        data = []
        for line in lines:
            x, y , label = line.split(' ')
            data.append([float(x), float(y), float(label)])
        return np.array(data, dtype='float32')


def plot_model(f_predict, dataset):
        n = 500
        x_min = np.min(dataset[:, 0])
        x_max = np.max(dataset[:, 0])
        y_min = np.min(dataset[:, 1])
        y_max = np.max(dataset[:, 1])

        x = np.linspace(x_min, x_max, n)
        y = np.linspace(y_min, y_max, n)
        xv, yv = np.meshgrid(x, y)
        z = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                item = np.array([[xv[i, j], yv[i, j]]])
                z[i, j] = f_predict(item)

        X = [ item[0] for item in dataset ]
        Y = [ item[1] for item in dataset ]
        L = [ item[2] for item in dataset ]

        plt.figure('Model Visualization')
        plt.contourf(xv, yv, z, cmap=plt.cm.Spectral)
        plt.scatter(X, Y, c=L, edgecolors='k', cmap=plt.cm.Spectral)
        plt.show()


if __name__ == '__main__':
    dataset = load_dataset()

    x = tf.placeholder(dtype='float32', shape=[None, 2])
    t = tf.placeholder(dtype='float32', shape=[None, 1])

    W1 = tf.get_variable('W1',
                         shape=[2, HIDDEN_UNIT],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.zeros([HIDDEN_UNIT]))
    W2 = tf.get_variable('W2',
                         shape=[HIDDEN_UNIT, 1],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.zeros([1]))

    y1 = tf.matmul(x, W1) + b1
    o1 = tf.tanh(y1)
    y2 = tf.matmul(o1, W2) + b2
    out = tf.sigmoid(y2)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y2))
    train = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)

    predict = tf.cast(tf.greater(out, 0.5), 'float32')
    correct_predict = tf.equal(predict, t)
    acc = tf.reduce_mean(tf.cast(correct_predict, 'float32'))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(EPOCH):
            sess.run(train, feed_dict={
                x: dataset[:, :2],
                t: dataset[:, 2:]
            })

        f_predict = lambda x_in: sess.run(predict, feed_dict={
            x: x_in
        })

        plot_model(f_predict, dataset)

