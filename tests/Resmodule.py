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