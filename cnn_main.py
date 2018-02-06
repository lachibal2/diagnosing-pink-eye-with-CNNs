#Dependancies:
import tensorflow as tf
import numpy as np
import os
import argparse

from util.print_progress import print_progress, showLossGraph
from util.change_to_array import changeToArray
from util.get_batches import createData, getBatches

#parsing console arguments:
"""
p = ArgumentParser()
p.add_argument('-n', '--name', required=True, help='name of file in "to_diagnose" to be diagnosed', type=str)
arguments = p.parse_args()

file_names = arguments.name.split(',') #parsing file names
"""
file_names = ['test1.jpg', 'test2.jpg']

class MisingTrainingDataError(Exception): #new error for missing training data
    def __init__(self):
        Exception.__init__('Missing Folder: "training images"')

if 'training images' not in os.listdir():
    raise MissingTrainingDataError

#Import DATA
imageList = createData()

#training parameters:
image_size = 128

learning_rate = 0.001
num_steps = 200
batch_size = 20
display_step = 10

num_input = 128 ** 2 #128x128 images are taken in by the cnn
num_classes = 2 #pink eye, or not pink eye
dropout = 0.75 #dropout probability

x = tf.placeholder(tf.float32, [None, num_input])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) #dropout probability

#Wrappers
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x) #relu activation function


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding="SAME")

#main
def CNN(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1,128,128,1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, dropout)
    return tf.add(tf.matmul(fc1, weights['out']), biases['out'])

weights = {
    'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64,1024])),
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

logits = CNN(x, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) #TODO: experiment with optemizers
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

loss_list = []
with tf.Session as sess:
    sess.run(init)

    for i in range(1, num_steps + 1):
        batch_x, batch_y = getBatches(imageList, batch_size, num_input, num_classes, image_size=image_size)#TODO: make batches
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if i % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            loss_list.append(loss)
            print_progress(i, loss, acc, num_steps + 1)
    print('Optimization is done!')
    
    for i in range(len(file_names)):
        current_file = changeToArray(file_names[i], i)
        result = sess.run(accuracy, feed_dict={X: 'TODO',Y: 'TODO',keep_prob: 1.0})
        print("Image '" + file_names[i] + "' has a {}% chance of having pink-eye".format(result))

    showLossGraph(loss_list, num_steps)
