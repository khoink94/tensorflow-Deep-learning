'''
A nearest neighbor learning algorithm example using TensorFlow library.
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# In this example, we limit mnist data

Xtr, Ytr = mnist.train.next_batch(5000) # 5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200) #200 for testing

# tf graph input

xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Nearest neighbor calculation using L1 distance
# calculte L1 distance

distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices = 1)

# Prediction: get min distance index (nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

#Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
#loop over test data
    for i in range(len(Xte)):
        #get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i,:] })
        #get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
              "true class:", np.argmax(Yte[i]))
        # Calculate accuracy
        print(Ytr.shape)
        print(Yte.shape)
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done")
    print("Accuracy:", accuracy)
