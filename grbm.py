# Gaussian Restrict Boltzmann Machine

import tensorflow as tf 
import math
import timeit
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images

class GRBM(object):
	# Gaussian Restrict Boltzmann Machine
	def __init__(self, inp = None, n_visible = 784, n_hidden = 500, W = None, hbias = None, vbias = None, sigma = 1):
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		if inp is None:
			inp = tf.placeholder(dtype = tf.float32, shape = [None, self.n_visible])
		self.input = inp
		if W is None:
			low = -4.0 * math.sqrt(6.0 / (n_visible + n_hidden))
			high = 4.0 * math.sqrt(6.0 / (n_visible + n_hidden))
			W = tf.Variable(tf.random_uniform([self.n_visible, self.n_hidden], minval = low, maxval = high, dtype = tf.float32))
		self.W = W
		if hbias is None:
			self.hbias = hbias
		self.hbias = hbias
		if vbias is None:
			self.vbias = vbias
		self.vbias = vbias
		self.sigma = sigma
		self.params = [self.W, self.hbias, self.vbias]


	def propup(self, visible):
		# This function propagates the visible units activation upwards to the hidden unit
		return tf.nn.sigmoid(tf.matmul(visible, self.W)/self.sigma **2 + self.hbias)	

	def propdown(self, hidden):
		# This function propagates the hidden units activaion downwards to the visible unit
		return (2 * math.pi* self.sigma ** 2) ** 0.5 tf.matmul(hidden, tf.transpose(self.W)) + self.vbias

	def sample_bernoulli(self, prob):
		return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))

	def sample_gaussian(self, x, sigma):
		return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)

	def sample_h_given_v(self, v0_sample):
		# This function infers state of hidden units given visible units
		# get a sample of the hiddens given their activation
		h1_mean = self.propup(v0_sample)
		h1_sample = self.sample_bernoulli(h1_mean)
		return (h1_mean, h1_sample)

	def sample_v_given_h(self, h0_sample):
		# This function infers state of visible units given hidden units
		# get a sample of the hiddens given their activation
		v1_mean = self.propdown(h0_sample)
		v1_sample = self.sample_gaussian(v1_sample)








