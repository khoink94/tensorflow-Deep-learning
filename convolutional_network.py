'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf 
#import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10

# network parameters
n_input = 784 #mnist data input
n_classes = 10 # mnist total classes
dropout = 0.75 # dropout, probability to keep units

#tf graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32) # dropout


# create some wrappers for simplicity
def conv2d(x, W, b, strides = 1):
	# conv2d wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides = [1,strides, strides, 1], padding = 'SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k =2):
# maxpool2d wrapper	
	return tf.nn.max_pool(x, ksize = [1, k , k, 1], strides =[ 1, k, k, 1], padding = 'SAME')

# create model
def conv_net(x, weights, biases, dropout):
	# reshape input picture
	x = tf.reshape(x, shape = [-1, 28, 28, 1])

	#convolution layer
	conv1 = conv2d(x , weights['wc1'], biases['bc1'])
	#max pooling (down sampling)
	conv1 = maxpool2d(conv1, k =2)

	#convolution layer
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	#max pooling(down sampling)
	conv2 = maxpool2d(conv2, k =2)

	#fully connected layer
	#reshape conv2 output to fit fully connected layer input

	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)

	#apply dropout
	fc1 = tf.nn.dropout(fc1, dropout)

	#output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

#store layers weight and bias

weights = {
	#5x5 conv, 1 input, 32 output
	'wc1' : tf.Variable(tf.random_normal([5,5,1,32])),
	#5x5 conv, 32 inputs, 64 outputs
	'wc2' : tf.Variable(tf.random_normal([5,5,32,64])),
	# fully connected, 7*7*64 inputs, 1024 outputs
	'wd1' : tf.Variable(tf.random_normal([7*7*64,1024])),
	# 1024 inputs, 10 outputs
	'out': tf.Variable(tf.random_normal([1024,n_classes]))
}

biases = {
	'bc1' : tf.Variable(tf.random_normal([32])),
	'bc2' : tf.Variable(tf.random_normal([64])),
	'bd1' : tf.Variable(tf.random_normal([1024])),
	'out' : tf.Variable(tf.random_normal([n_classes]))
}

#construct model
pred = conv_net(x, weights, biases, keep_prob)

#define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#Evaluate model

correct_pre = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

#initializing the variables
init = tf.global_variables_initializer()

#launch the graph
with tf.Session() as sess:
	sess.run(init)
	step =1
	#keep training until reach max iterations
	while step * batch_size < training_iters:
		batch_x, batch_y = mnist.train.next_batch(batch_size)

		#run optimization op (backprop)
		sess.run(optimizer, feed_dict = {x :batch_x, y:batch_y, keep_prob: dropout})

		if step % display_step == 0:
			# calculate batch loss and accuracy
			loss, acc = sess.run([cost,accuracy], feed_dict ={x: batch_x, y :batch_y, keep_prob : 1.})

			print("iter", str(step*batch_size), " minibatch loss =" , "{:.6f}".format(loss) ," Training Accuracy =" ,"{:.5f}".format(acc))

		step +=1
	print("optimization finished")

	#Calculate accuracy for 256 mnist test images
	print ("testing accuracy:",sess.run(accuracy, feed_dict = {x: mnist.test.images[:256], y:mnist.test.labels[:256], keep_prob: 1.}))