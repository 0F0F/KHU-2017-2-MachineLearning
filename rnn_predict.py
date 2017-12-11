import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os



class predictor:
    def __init__(self, model_path, sequence_length, hidden_dimension, scale_base, scale_factor, trial_num=None):
        self.scale_factor = scale_factor
        self.scale_base = scale_base

        self.sequence_length = sequence_length
        self.hidden_dimension = hidden_dimension

        self.data_dimension = 1
        self.output_dimension = 1

        self.X = tf.placeholder(tf.float32, [None, sequence_length, self.data_dimension])
        self.Y = tf.placeholder(tf.float32, [None, self.output_dimension])

        self.cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=hidden_dimension, state_is_tuple=True, activation=tf.tanh)
        self.outputs, self.states = tf.nn.dynamic_rnn(self.cell, self.X, dtype=tf.float32)
        self.Y_pred = tf.contrib.layers.fully_connected(
            self.outputs[:, -1], self.output_dimension, activation_fn=None)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(self.sess, model_path)

        self.dataX = [[]]

    def scale(self, datum):
        return (datum - self.scale_base) / self.scale_factor

    def scale_back(self, datum):
        return (datum * self.scale_factor) + self.scale_base

    def renew_data(self, datum):
        scaled_datum = self.scale(datum)
        if len(self.dataX[0]) >= self.sequence_length:
            del self.dataX[0][0]

        self.dataX[0].append([scaled_datum])

    def predict(self):
        if len(self.dataX[0]) != self.sequence_length:
            return 0

        else:
            predict = self.sess.run(self.Y_pred, feed_dict = {self.X: self.dataX[0:1]})
            predict = self.scale_back(np.array(predict)[0])

            return predict[0]
