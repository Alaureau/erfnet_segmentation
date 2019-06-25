def factorized_res_moduleOLD(x, is_training, dropout=0.3, dilation=1, name="fres"):
    with arg_scope(get_conv_arg_scope(is_training=is_training, bn=True)):
        with tf.variable_scope(name):
            n_filters = x.shape.as_list()[-1]
            y = conv(x, num_outputs=n_filters, kernel_size=[3,1], normalizer_fn=None, scope="conv_a_3x1")
            y = conv(y, num_outputs=n_filters, kernel_size=[1,3], scope="conv_a_1x3")
            y = conv(y, num_outputs=n_filters, kernel_size=[3,1], rate=dilation, normalizer_fn=None, scope="conv_b_3x1")
            y = conv(y, num_outputs=n_filters, kernel_size=[1,3], rate=dilation, scope="conv_b_1x3")
            y = dropout_layer(y, rate=dropout)
            y = tf.add(x,y, name="add")
    print("DEBUG: {} {}".format(name, y.shape.as_list()))
    return y


def factorized_res_module(x, is_training, dropout=0.3, dilation=[1,1], l2=None, name="fres"):
    reg = None if l2 is None else l2_regularizer(l2)
    with arg_scope(get_conv_arg_scope(reg=reg, is_training=is_training, bn=True)):
        with tf.variable_scope(name):
            n_filters = x.shape.as_list()[-1]
            y = conv(x, num_outputs=n_filters, kernel_size=[3,1], rate=dilation[0], normalizer_fn=None, scope="conv_a_3x1")
            y = conv(y, num_outputs=n_filters, kernel_size=[1,3], rate=dilation[0], scope="conv_a_1x3")
            y = conv(y, num_outputs=n_filters, kernel_size=[3,1], rate=dilation[1], normalizer_fn=None, scope="conv_b_3x1")
            y = conv(y, num_outputs=n_filters, kernel_size=[1,3], rate=dilation[1], scope="conv_b_1x3")
            y = dropout_layer(y, rate=dropout)
            y = tf.add(x,y, name="add")
    print("DEBUG: {} {}".format(name, y.shape.as_list()))
    print("DEBUG: L2 in factorized res module {}".format(l2))
    return y
    
class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

#https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
tf.nn.atrous_conv2d(
    value,
    filters,
    rate,
    padding,
    name=None
)

tf.nn.conv2d(
    input,
    filter=None,
    strides=None,
    padding=None,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None,
    filters=None
)
tf.contrib.layers.conv2d(
    inputs,
    num_outputs,
    kernel_size,
    stride=1,
    padding='SAME',
    data_format=None,
    rate=1,
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
)
