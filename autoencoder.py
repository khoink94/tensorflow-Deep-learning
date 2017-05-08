
""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

#import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

#Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

#network parameters
n_hidden_1 = 256 
n_hidden_2 = 128
n_input = 784

#tf graph input
X = tf.placeholder("float",[None, n_input])

weights = {
	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
	'encoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_h2': tf.Variable(tf.random_normal([n_input]))
}

# Building the encoder
def encoder(x):
	# encoder hidden layer with sigmoid activtion
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_h1']))
	#encoder hidden layer with sigmoid activation 
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_h2']))

	return layer_2
#building decoder
def decoder(x):
	#decoder hidden layer with sigmoid activation
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_h1']))
	#decoder hidden layer with sigmoid activation
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_h2']))
	return layer_2

#construct the model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

#prediction
y_pred = decoder_op
#targets are inout data
y_true = X

#define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#initializing the variables
init = tf.global_variables_initializer()

#launch the graph

with tf.Session() as sess:
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)
	# training cyccle
	for epoch in range(training_epochs):
		#loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)

			#run optimization op (backprop) and cost op (get loss value)
			_, c = sess.run([optimizer, cost], feed_dict = {X: batch_xs})

		#display logs per epoch step
		if (epoch +1) % display_step ==0:
			print("epoch", '%04d' % (epoch +1), "cost", "{:.9f}".format(c))

	print("optimization finished")

	#applying encode and decode over test set
	encode_decode = sess.run(y_pred, feed_dict = {X : mnist.test.images[:examples_to_show]})
	#compare original images with their reconstructions
	f,a = plt.subplots(2, 10, figsize= (10,2))
	for i in range(examples_to_show):
		a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
		a[1][i].imshow(np.reshape(encode_decode[i], [28,28]))
	f.show()
	plt.draw()
	plt.waitforbuttonpress()