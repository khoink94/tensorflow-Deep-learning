from __future__ import print_function
import tensorflow as tf

hello = tf.constant('Hello World')
# start tf session
sess = tf.Session()

print(sess.run(hello))
