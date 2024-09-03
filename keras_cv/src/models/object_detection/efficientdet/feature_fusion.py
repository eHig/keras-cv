import keras


from keras.src.activations import softmax
from keras_cv.src.backend import ops

class FeatureFusion(keras.layers.Layer): #todo handle channel-first
    def __init__(self, weight_method='attn', data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        self.weight_method = weight_method

    def build(self, input_shape):
        num_inputs = len(input_shape)
        assert(num_inputs > 1)
        for shape in input_shape:
            assert(shape == input_shape[0])
        num_filters = input_shape[0][-1]

        if self.weight_method == 'attn':
            self.call_method = self._attn
            self._initialize_weights(num_inputs)
        elif self.weight_method == 'fastattn':
            self.call_method = self._fastattn
            self._initialize_weights(num_inputs)
        elif self.weight_method == 'channel_attn':
            self.call_method = self._channel_attn
            self._initialize_weights(num_inputs, num_filters)
        elif self.weight_method == 'channel_fastattn':
            self.call_method = self._channel_fastattn
            self._initialize_weights(num_inputs, num_filters)
        elif self.weight_method == 'sum':
            self.call_method = self._sum
        else:
            raise ValueError(f'Unknown weight_method {self.weight_method}')

    def _attn(self, inputs):
        normalized_weights = softmax(self.weightings)
        inputs_stacked = ops.stack(inputs, axis=-1)
        return ops.sum(normalized_weights*inputs_stacked,  axis=-1)

    def _fastattn(self, inputs):
        edge_weights = ops.relu(self.weightings)
        weights_sum = ops.sum(edge_weights)
        normalized_weights = edge_weights / (weights_sum + 0.0001)
        weighted_inputs = [inputs[i] * normalized_weights[0][i] for i in range(len(inputs))]
        res = ops.sum(ops.stack(weighted_inputs, axis=0), axis=0)
        return res

    def _channel_attn(self, inputs):
        normalized_weights = softmax(self.weightings, axis=-1)
        inputs_stacked = ops.stack(inputs, axis=-1)
        return ops.sum(inputs_stacked * normalized_weights, axis=-1)

    def _channel_fastattn(self, inputs):
        edge_weights = ops.relu(self.weightings)
        weights_sum = ops.sum(edge_weights, axis=1, keepdims=True)
        normalized_weights = edge_weights / (weights_sum + 0.0001)
        weighted_inputs = [inputs[i] * normalized_weights[:, i] for i in range(len(inputs))]
        return ops.sum(ops.stack(weighted_inputs, axis=0), axis=0)

    def _sum(self, inputs):
        return ops.sum(inputs, axis=0)

    def _initialize_weights(self, num_inputs, width=1):
        shape = (width, num_inputs)
        # print(shape)
        self.weightings = self.add_weight(name=f'weights',
                            shape=shape,
                            initializer="ones",
                            trainable=True)


    def call(self, inputs):
        return self.call_method(inputs)

    def get_config(self):
        config = super(FeatureFusion, self).get_config()
        config.update({'weight_method': self.weight_method})
        return config


