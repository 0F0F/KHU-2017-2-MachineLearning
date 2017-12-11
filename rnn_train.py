import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
import getopt
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def MinMaxScaler(data):
    min_term = np.min(data, 0)
    max_term = np.max(data, 0)
    numerator = data - min_term 
    denominator = max_term - min_term + 1e-7
    return numerator / denominator, denominator, min_term

def scale_back(data, denominator, min_term):
    return data * denominator + min_term


def progress_bar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{2}/{3}: [{0}] {1:.8f}%".format(arrow + spaces, 100. * percent, value, endvalue))
    sys.stdout.flush()



def train(train_data, sequence_length, data_dimension, hidden_dimension, output_dimension, learning_rate, iterations, trial_count):
        # Open, High, Low, Volume, Close
    trial_info = '{}_{}_{}'.format(train_data, sequence_length, hidden_dimension)
    file_path = './{}_log.csv'.format(train_data)
    save_to = './{}.ckpt'.format(trial_info)

    xy = np.loadtxt(file_path, delimiter=',')
    xy, den, min_term = MinMaxScaler(xy)
    x = [[f] for f in xy]
    y = x[0:]

    dataX = []
    dataY = []
    for i in range(0, len(y) - sequence_length):
        _x = x[i:i + sequence_length]
        _y = y[i + sequence_length]
        dataX.append(_x)
        dataY.append(_y)


    train_size = int(len(dataY) * 0.7)
    trainX, testX = np.array(dataX[0:train_size]), np.array(
        dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(
        dataY[train_size:len(dataY)])

    X = tf.placeholder(tf.float32, [None, sequence_length, data_dimension])
    Y = tf.placeholder(tf.float32, [None, 1])

    cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_dimension, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(
        outputs[:, -1], output_dimension, activation_fn=None)

    loss = tf.reduce_sum(tf.square(Y_pred - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("trial#   {}: {}".format(trial_count, trial_info))
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={
                                    X: trainX, Y: trainY})

        test_predict = sess.run(Y_pred, feed_dict={X: testX})
        rmse_val = sess.run(rmse, feed_dict={
                        targets: testY, predictions: test_predict})
        print("RMSE: {}".format(rmse_val))
        save_path = saver.save(sess, save_to)
        print("model saved at.. {}".format(save_path))
        return rmse_val



def demo():
    train_data = '1y'
    sequence_length = 3  # varies
    hidden_dimension = 6

    data_dimension = 1
    output_dimension = 1
    learning_rate = 0.01
    iterations = 10000

    trial_info = '{}_{}_{}'.format(train_data, sequence_length, hidden_dimension)
    file_path = './{}_log.csv'.format(train_data)
    save_to = './{}.ckpt'.format(trial_info)

    xy = np.loadtxt(file_path, delimiter=',')
    xy, den, min_term = MinMaxScaler(xy)
    x = [[f] for f in xy]
    y = x[0:]

    dataX = []
    dataY = []
    for i in range(0, len(y) - sequence_length):
        _x = x[i:i + sequence_length]
        _y = y[i + sequence_length]
        dataX.append(_x)
        dataY.append(_y)


    train_size = int(len(dataY) * 0.7)
    trainX, testX = np.array(dataX[0:train_size]), np.array(
        dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(
        dataY[train_size:len(dataY)])

    X = tf.placeholder(tf.float32, [None, sequence_length, data_dimension])
    Y = tf.placeholder(tf.float32, [None, 1])

    cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_dimension, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(
        outputs[:, -1], output_dimension, activation_fn=None)

    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={
                                    X: trainX, Y: trainY})
            progress_bar(i, iterations)


        print('\n')

        test_predict = sess.run(Y_pred, feed_dict={X: testX})
        rmse_val = sess.run(rmse, feed_dict={
                        targets: testY, predictions: test_predict})
        print("RMSE: {}".format(rmse_val))
        save_path = saver.save(sess, save_to)
        print("model saved at.. {}".format(save_path))

        plt.plot(testY)
        plt.plot(test_predict)
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")

    plt.show()


if __name__ == "__main__":
    opts, atgs = getopt.getopt(sys.argv[1:], 's:h:r:c:d')
    sequence_length = 7
    hidden_dimension = 3
    resource = '6m'
    count = 0

    data_dimension = 1
    output_dimension = 1
    learning_rate = 0.01
    iterations = 500

    for opt, arg in opts:
        if opt == '-s':
            sequence_length = int(arg)
        if opt == '-h':
            hidden_dimension = int(arg)
        if opt == '-r':
            resource = arg
        if opt == '-c':
            count = int(arg)
        if opt == '-d':
            print('demo')
            demo()
            exit(-1)


    cost = train(resource, sequence_length, data_dimension, hidden_dimension, output_dimension, learning_rate, iterations, count)
    with open('train_log.csv', 'a') as log:
        wrt = [resource, sequence_length, hidden_dimension, cost]
        csv.writer(log).writerow(wrt[0:])

    exit(cost)

print('not executed as __main__')
exit(-1)



