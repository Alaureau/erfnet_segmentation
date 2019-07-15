from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

print("Tensorflow version =",tf.__version__)
print("Keras version =",tf.keras.__version__)

class non_bottleneck_1d (Model):
    def __init__(self, chann, dropprob, dilated):        
        super(non_bottleneck_1d,self).__init__()
        self.conv3x1_1 = layers.Conv2D(filters=chann,kernel_size=(3, 1),strides=1,padding="same", use_bias=True)
        self.conv1x3_1 = layers.Conv2D(filters=chann,kernel_size=(1, 3),strides=1,padding="same", use_bias=True)
        self.bn1 = layers.BatchNormalization(epsilon=1e-3)
        self.conv3x1_2 = layers.Conv2D(filters=chann,kernel_size=(3, 1),strides=1,padding="same", use_bias=True, dilation_rate=(dilated,1))
        self.conv1x3_2 = layers.Conv2D(filters=chann,kernel_size=(1, 3),strides=1,padding="same", use_bias=True, dilation_rate=(1, dilated))
        self.bn2 = layers.BatchNormalization(epsilon=1e-3)
        self.dropout = layers.Dropout(dropprob)
    def  call(self, input, training=False):
        y = self.conv3x1_1(input)
        y = layers.ReLU()(y)
        y = self.conv1x3_1(y)
        y = self.bn1(y)
        y = layers.ReLU()(y)
        y = self.conv3x1_2(y)
        y = layers.ReLU()(y)
        y = self.conv1x3_2(y)
        y = self.bn2(y)
        if training:
            y = self.dropout(y)
        y = layers.add([input, y])
        return layers.ReLU()(y)

encoder_input = tf.keras.Input(shape=(1280, 720, 3), name='original_img')
#Down sampler block 3,16
a = layers.Conv2D(filters=16-3,kernel_size=3,strides=2,padding="same", use_bias=True)(encoder_input)
b = layers.MaxPool2D(pool_size = 2,strides = 2,padding = "same")(encoder_input)
y = layers.concatenate([a, b],axis=-1)
y = layers.BatchNormalization(axis = -1, epsilon = 1e-03)(y)
y = layers.ReLU()(y)
#Down sampler block 16, 64
c = layers.Conv2D(filters=64-16,kernel_size=3,strides=2,padding="same", use_bias=True)(y)
d = layers.MaxPool2D(pool_size = 2,strides = 2,padding = "same")(y)
y = layers.concatenate([a, b],axis=-1)
y = layers.BatchNormalization(axis = -1, epsilon = 1e-03)(y)
y = layers.ReLU()(y)
#Non bottleneck 64
for i in range (0,5):
    y = non_bottleneck_1d(64, 0.03, 1)(y)
#Down sampler block 64, 128
e = layers.Conv2D(filters=128-64,kernel_size=3,strides=2,padding="same", use_bias=True)(y)
f = layers.MaxPool2D(pool_size = 2,strides = 2,padding = "same")(y)
y = layers.concatenate([a, b],axis=-1)
y = layers.BatchNormalization(axis = -1, epsilon = 1e-03)(y)
y = layers.ReLU()(y)
#Non bottleneck 128
for i in range (0,2):
    y = non_bottleneck_1d(128, 0.3, 2)(y)
    y = non_bottleneck_1d(128, 0.3, 4)(y)
    y = non_bottleneck_1d(128, 0.3, 8)(y)
    y = non_bottleneck_1d(128, 0.3, 16)(y)
encoder_output = y
encoder = tf.keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()
tf.keras.utils.plot_model(encoder, 'encoder.png', show_shapes=True, expand_nested=True)

decoder_input = tf.keras.Input(shape= (encoder.output_shape[1],encoder.output_shape[2], encoder.output_shape[3]), name='filtered_img')
#Upsampler Block 64
y = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", output_padding=1, use_bias=True)(decoder_input)
y = layers.BatchNormalization(epsilon=1e-3)(y)
#Non bottleneck 64, 1 dilation
y = non_bottleneck_1d(64, 0.03, 1)(y)
y = non_bottleneck_1d(64, 0.03, 1)(y)
#Upsampler Block 16
y = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same", output_padding=1, use_bias=True)(y)
y = layers.BatchNormalization(epsilon=1e-3)(y)
y = layers.ReLU()(y)
#Non bottleneck 16, 1 dilation
y = non_bottleneck_1d(16, 0.03, 1)(y)
y = non_bottleneck_1d(16, 0.03, 1)(y)
#Upsampler Block Nb_classes
y = layers.Conv2DTranspose(filters=4, kernel_size=3, strides=2, padding="same", output_padding=1, use_bias=True)(y)
y = layers.BatchNormalization(epsilon=1e-3)(y)
y = layers.ReLU()(y)

decoder_output = y
decoder = tf.keras.Model(decoder_input, decoder_output, name='decoder')
decoder.summary()
tf.keras.utils.plot_model(decoder, 'decoder.png', show_shapes=True, expand_nested=True)