import numpy as np

import keras_cv.src.layers
from keras_cv.src.backend import keras
#
# SEPARABLE_CONV_KERNEL_INITIALIZER = {
#     "class_name": "VarianceScaling",
#     "config": {
#         "scale": 2.0,
#         "mode": "fan_out",
#         "distribution": "truncated_normal",
#     },
# }

CONV_KERNEL_INITIALIZER = {
    "class_name": "RandomNormal",
    "config": {
        "mean": 0.0,
        "stddev": 0.01,
        "distribution": "truncated_normal",
    },
}

CLASS_BIAS_INITIALIZER = {
    "class_name": "CONSTANT",
    "config": {
        "value": -np.log((1 - 0.01) / 0.01)
    }
}



class BoxNet(keras.layers.Layer):
    """Box regression network."""

    def __init__(self,
                 num_anchors=9,
                 num_filters=32,
                 act_type='swish',
                 repeats=4,
                 separable_conv=True,
                 survival_prob=None,
                 data_format='channels_last',
                 name='box_net',
                 **kwargs):
        """Initialize BoxNet.

        Args:
          num_anchors: number of  anchors used.
          num_filters: number of filters for "intermediate" layers.
          act_type: String of the activation used.
          repeats: number of "intermediate" layers.
          separable_conv: True to use separable_conv instead of conv2D.
          survival_prob: if a value is set then drop connect will be used.
          name: Name of the layer.
            head).
          **kwargs: other parameters.
        """

        super().__init__(name=name, **kwargs)

        # print(f"box pred {num_anchors=}")

        self.num_anchors = num_anchors
        self.num_filters = num_filters
        self.repeats = repeats
        self.separable_conv = separable_conv
        self.data_format = data_format

        self.activation_function = keras.activations.get(act_type) if act_type else None
        self.stochastic_depth = keras_cv.src.layers.StochasticDepth(rate=1.0 - survival_prob) if survival_prob else None

        self.conv_ops = []
        self.bns = []
        #todo feature_only, grad_checkpoint
        for i in range(self.repeats):
            self.conv_ops.append(self.conv(f"box-{i}", num_filters))

        self.boxes_layer = self.conv("box-predict", num_anchors*4)



    def build(self, input_shapes):
        self.bns = []
        for i in range(self.repeats):
            bn_per_level = []
            for level in range(len(input_shapes)):
                bn_per_level.append(
                    keras.layers.BatchNormalization(axis=-1 if self.data_format == "channels_last" else 1)
                )
            self.bns.append(bn_per_level)


    #todo static
    def conv(self, name, num_filters):
        if self.separable_conv:
            return keras.layers.SeparableConv2D(
                filters=num_filters,
                depth_multiplier=1,
                pointwise_initializer="VarianceScaling",
                depthwise_initializer="VarianceScaling",
                bias_initializer="zeros",
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                padding="same",
                name=name)
        else:
            return keras.layers.Conv2D(
                filters=num_filters,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                bias_initializer="zeros",
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                padding='same',
                name=name)


    #todo feature_only needed?, recompute_grad needed?
    def call(self, inputs):
        box_outputs = []
        for level in range(len(inputs)):
            x = inputs[level]
            for i in range(self.repeats):
                x = self.conv_ops[i](x)
                x = self.bns[i][level](x)
                if self.activation_function:
                    x = self.activation_function(x)
                if i>0 and self.stochastic_depth:
                    x = self.stochastic_depth([inputs[level],x])
            box_outputs.append(self.boxes_layer(x))
        shapes = [o.shape for o in box_outputs]
        # print(f"box head outputs: {shapes}")
        return box_outputs

class ClassNet(keras.layers.Layer):
    """Box regression network."""

    def __init__(self,
                 num_classes=90,
                 num_anchors=9,
                 num_filters=32,
                 act_type='swish',
                 repeats=4,
                 separable_conv=True,
                 survival_prob=None,
                 data_format='channels_last',
                 name='class_net',
                 **kwargs):
        """Initialize ClassNet.

        Args:
          num_anchors: number of  anchors used.
          num_filters: number of filters for "intermediate" layers.
          act_type: String of the activation used.
          repeats: number of "intermediate" layers.
          separable_conv: True to use separable_conv instead of conv2D.
          survival_prob: if a value is set then drop connect will be used.
          name: Name of the layer.
            head).
          **kwargs: other parameters.
        """

        super().__init__(name=name, **kwargs)

        self.num_anchors = num_anchors
        self.num_filters = num_filters
        self.repeats = repeats
        self.separable_conv = separable_conv
        self.data_format = data_format

        self.activation_function = keras.activations.get(act_type) if act_type else None
        self.stochastic_depth = keras_cv.src.layers.StochasticDepth(rate=1.0 - survival_prob) if survival_prob else None

        self.conv_ops = []
        self.bns = []
        #todo feature_only, grad_checkpoint
        for i in range(self.repeats):
            self.conv_ops.append(self.conv(f"class-{i}", num_filters))
        initializer = keras.initializers.Constant(
            -np.log((1 - 0.01) / 0.01)
        )
        self.class_layer = self.conv("class-predict", num_anchors*num_classes, bias_initializer=initializer)

    def build(self, input_shapes):
        self.bns = []
        for i in range(self.repeats):
            bn_per_level = []
            for level in range(len(input_shapes)):
                bn_per_level.append(
                    keras.layers.BatchNormalization(axis=-1 if self.data_format == "channels_last" else 1) #todo name)
                )
            self.bns.append(bn_per_level)

    #todo static
    def conv(self, name, num_filters, bias_initializer="zeros"):
        if self.separable_conv:
            return keras.layers.SeparableConv2D(
                filters=num_filters,
                depth_multiplier=1,
                pointwise_initializer="VarianceScaling",
                depthwise_initializer="VarianceScaling",
                bias_initializer=bias_initializer,
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                padding="same",
                name=name)
        else:
            return keras.layers.Conv2D(
                filters=num_filters,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                bias_initializer=bias_initializer,
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                padding='same',
                name=name)

    #todo feature_only needed?, recompute_grad needed?
    def call(self, inputs):
        class_outputs = []
        for level in range(len(inputs)):
            x = inputs[level]
            for i in range(self.repeats):
                x = self.conv_ops[i](x)
                x = self.bns[i][level](x)
                if self.activation_function:
                    x = self.activation_function(x)
                if i > 0 and self.stochastic_depth:
                    x = self.stochastic_depth([inputs[level], x])
            class_outputs.append(self.class_layer(x))
        return class_outputs


