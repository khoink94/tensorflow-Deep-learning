'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.

'''

from __future__ import print_function

import tensorflow as tf 
from tensorflow.contrib import rnn

#import Mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# to classify images using a recurrent neural network, we consider every image
# row as a sequence of pixels. Because mnist image shape is 28*28px, 
# we will then handle 28 sequences of 28 steps for every sample.

# parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

#network parameters
n_input = 28 #mnist data input (img shape 28*28)
n_steps = 28 #timesteps
n_hidden = 128
n_classes = 10 

#tf graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

#Define weights
weights = {
	'out' : tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
	'out' : tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
	# prepare data shpe to match 'rnn' function requirements
	# current data input shape : (batch_size, n_steps, n_input)
	#required shape : 'n_steps' tensors list of shpe(batch_size, n_input)

	#unstack to get a list of n_steps tensors of shape (batch_size, n_input)

	x = tf.unstack(x, n_steps, 1)

	#define a lstm cell with tensorflow
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
	#get lstm cell utput
	output, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)

	#linear activation, using rnn inner loop last output
	return tf.matmul(output[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

#Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#initializing the variables
init = tf.global_variables_initializer()

#launch the graph
with tf.Session() as sess:
	sess.run(init)
	step = 1
	# keep training until reach max iterations
	while step * batch_size < training_iters:
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		#reshape data to get 28 squences of 28 elements
		batch_x = batch_x.reshape((batch_size, n_steps, n_input))
		#run optimization op (backprop)
		sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
		if step % display_step == 0 :
			#calculate batch accuracy
			acc = sess.run(accuracy, feed_dict = {x : batch_x, y: batch_y})
			#loss
			loss = sess.run(cost, feed_dict = {x :batch_x,y: batch_y})
			print("iter", str(step*batch_size), "minibatch loss", "{:.6f}".format(loss),"training accuracy =","{:.5f}".format(acc))

		step += 1
	print("optimization finished")

	#calculate accuracy for 128 mnist test images
	test_len = 128
	test_data = mnist.test.images[:test_len].reshape((-1, n_steps,n_input))
	test_label = mnist.test.labels[:test_len]
	print("testing accuracy:", sess.run(accuracy, feed_dict = {x:test_data, y:test_label}))
