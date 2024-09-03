import functools

import keras


class OpAfterCombine(keras.layers.Layer):
    """Operation after combining input features during feature fusion."""

    def __init__(self,
                 conv_bn_act_pattern,
                 separable_conv,
                 num_filters,
                 activation,
                 data_format,
                 name='op_after_combine'):
        super().__init__(name=name)
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.separable_conv = separable_conv
        self.num_filters = num_filters
        self.data_format = data_format
        if self.separable_conv:
            conv2d_layer = functools.partial(
                keras.layers.SeparableConv2D, depth_multiplier=1)
        else:
            conv2d_layer = keras.layers.Conv2D

        self.conv_op = conv2d_layer(
            filters=num_filters,
            kernel_size=(3, 3),
            padding='same',
            use_bias=not self.conv_bn_act_pattern,
            data_format=self.data_format,
            name='conv')
        self.batch_norm = keras.layers.BatchNormalization(
            axis=-1 if self.data_format == "channels_last" else 1
        )
        self.activation_function = keras.activations.get(activation)

    def call(self, x):
        if not self.conv_bn_act_pattern:
            x = self.activation_function(x)
        x = self.conv_op(x)
        x = self.batch_norm(x)
        if self.conv_bn_act_pattern:
            x = self.activation_function(x)
        return x
