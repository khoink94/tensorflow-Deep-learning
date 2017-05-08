from __future__ import print_function

import tensorflow as tf

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.

a = tf.constant(2)
b = tf.constant(3)

# launch the default graph
with tf.Session() as sess:
    print(sess.run(a))
    print("a = 2, b = 3")
    print("Addition with constants: %i"%sess.run(a + b))
    print("Multiplication with constants:%i" % sess.run(a*b))

# Basic operations with variable as graph input
# The value returned by the constructor represents the output of the variable op.
# tf graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess:
    print("Addition with variables: %i" % sess.run(add, feed_dict={a:3, b:4}))
    print("Multiplication with variables: %i" %sess.run(mul, feed_dict={a:4, b:5}))

# ----------------
# More in details:
# Matrix Multiplication from TensorFlow official tutorial

# Create a constant op that produces a 1x2 matrix. the op is added as a node to the default graph.
# The value returned by the cnstructor represents the output of the constnt op.

matrix1 = tf.constant([[3., 3.]])

# Create another constant that produces a 2x1 matrix
matrix2 = tf.constant([[2.],[ 2.]])

# Create a matmul op that takes matrix1 and matrix 2 as inputs, the returned value product represents the result of the matrix multiplication

product = tf.matmul(matrix1,matrix2)


# To run the matmul op we call the session run() method, passing product which represents the output of the matmul op. This indicates to the call that we want to get the output to the matmul op back

# all inputs needed by the op are run automatically by the session. ther typically are run in parallel.

# the call run product thus causes the execution of threes op in the graph: the two constants and matmul

#the output of the op is returned in result as a numpy ndarray object.

with tf.Session() as sess:

    result = sess.run(product)
    print(result)
