# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

# ERFNet implementation for Tensorflow
# Sept 2017
# Ronny RestRepo
#######################

# ERFNet Road full model definition for Pytorch
# May 2019
# Fabvio Pizzati and Fernando Garcı́a
#######################

from __future__ import print_function, division, unicode_literals
import numpy as np
import tensorflow as tf

#USEFUL LAYERS
conv = tf.nn.conv2d
batchnorm = tf.nn.batch_normalization
dropout = tf.nn.dropout
maxpool = tf.nn.max_pool2d
relu = tf.nn.relu

def DownsamplerBlock (x, n_filters, is_training, bn=False, use_relu=False, l2=None, name="down"):
	branch_a = conv(
		x,
	    filter=[3,1,n_filters,2*n_filters],
	    #TODO scope variables
	    strides=2,
	    padding=1,
	    #use_cudnn_on_gpu=True,
	    #data_format='NHWC',
	    #end of scope
	    #dilations=1,
	    name="conv_3x3"
	)
	branch_b = maxpool(
	    x,
	    ksize = 2,
	    strides = 2,
	    padding = "VALID",
	    #data_format='NHWC',
	    name="mpool"
	)
	y = tf.concat(
		[branch_a, branch_b],
		axis=-1,
		name="concat"
	)
	y = batchnorm(
	    y,
	    #mean,
	    #variance,
	    #offset,
	    #scale,
	    variance_epsilon = 1e-03,
	    name="bn_b"
	)
	y = relu(
		y,
		name = "relu"
	)
	return y

def non_bottleneck_1d(x, is_training, dropprob=0.3, dilation=1, name="fres"):
	y = conv(
		x,
	    filter=[3,1,n_filters,n_filters],
	    #TODO scope variables
	    #strides=1,
	    #padding="SAME",
	    #use_cudnn_on_gpu=True,
	    #data_format='NHWC',
	    #end of scope
	    #dilations=1,
	    name="conv_a_3x1"
	)
	y = relu(
		y,
		name = "relu_a"
	)
	y = conv(
		y,
	    filter=[1,3,n_filters,n_filters],
	    #TODO scope variables
	    #strides=1,
	    #padding="SAME",
	    #use_cudnn_on_gpu=True,
	    #data_format='NHWC',
		#end of scope
	    #dilations=1,
	    name="conv_a_1x3"
	) 
	y = batchnorm(
	    y,
	    #mean,
	    #variance,
	    #offset,
	    #scale,
	    variance_epsilon = 1e-03,
	    name="bn_a"
	)
	y = relu(
		y,
		name = "relu_b"
	)
	y = conv(
		y,
	    filter=[3,1,n_filters,n_filters],
	    #TODO scope variables
	    #strides=1,
	    #padding="SAME",
	    #use_cudnn_on_gpu=True,
	    #data_format='NHWC',
	    #end of scope
	    dilations=dilation,
	    name="conv_b_3x1"
	)
	y = relu(
		y,
		name = "relu_c"
	) 
	y = conv(
		y,
	    filter=[1,3,n_filters,n_filters],
	    #TODO scope variables
	    #strides=1,
	    #padding="SAME",
	    #use_cudnn_on_gpu=True,
	    #data_format='NHWC',
	    #end of scope
	    dilations=dilation,
	    name="conv_b_1x3"
	) 
	y = batchnorm(
	    y,
	    #mean,
	    #variance,
	    #offset,
	    #scale,
	    variance_epsilon = 1e-03,
	    name="bn_b"
	)
	y = dropout(
	    x,
	    #noise_shape=None,
	    #seed=None,
	    #name=None,
	    rate=dropprob
	)
	y = relu(
		y,
		name = "relu_d"
	)
	return y