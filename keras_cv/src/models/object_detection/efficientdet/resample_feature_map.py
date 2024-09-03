import keras


#todo move to layers

class ResampleFeatureMap(keras.layers.Layer):

    def __init__(self,
                 target_num_channels,
                 ratio,
                 downsampling,
                 apply_batchnorm=False,
                 conv_after_downsample=False,
                 data_format=None,
                 pooling_type=None,
                 upsampling_type=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_num_channels = target_num_channels
        self.apply_batchnorm = apply_batchnorm
        self.data_format = data_format
        self.conv2d = None

        downsampling = downsampling and ratio > 1
        self.resample = self.build_downsample(ratio, pooling_type) if downsampling else self.build_upsample(ratio, upsampling_type)
        self.conv_after_resample = downsampling and conv_after_downsample

    def build(self, input_shape):
        # print(f"building: {input_shape=}")
        if self.get_channels(input_shape) != self.target_num_channels:
            self.conv2d = keras.layers.Conv2D(
                self.target_num_channels, (1, 1),
                padding='same',
                data_format=self.data_format,
                name=f'{self.name}-conv2d',
            )
            if self.apply_batchnorm:
                channel_axis = -1 if self.data_format == "channels_last" else 1
                self.batch_norm = keras.layers.BatchNormalization(
                    axis=channel_axis
                )

    def _maybe_apply_1x1(self, feat):
        if self.conv2d:
            feat = self.conv2d(feat)
            if self.apply_batchnorm:
                feat = self.batch_norm(feat)
        return feat

    def get_channels(self, shape):
        # print(shape)
        shape[1] if self.data_format == 'channels_first' else shape[3]

    def call(self, feat):
        in_shape = feat.shape
        # print(f"{feat.shape=}")
        if not self.conv_after_resample:
            feat = self._maybe_apply_1x1(feat)
        feat = self.resample(feat)
        if self.conv_after_resample:
            feat = self._maybe_apply_1x1(feat)

        # print(f"{self.name=} - {type(self.resample)} - {in_shape} -> {feat.shape}")
        return feat

    def build_downsample(self, ratio: int, pooling_type: str):
        # print("build downsample")
        if pooling_type == 'max':
            return keras.layers.MaxPooling2D(
                pool_size=[ratio + 1, ratio + 1],
                strides=[ratio, ratio],
                padding='SAME',
                data_format=self.data_format)
        elif pooling_type == 'avg':
            return keras.layers.AveragePooling2D(
                pool_size=[ratio + 1, ratio + 1],
                strides=[ratio, ratio],
                padding='SAME',
                data_format=self.data_format)
        else:
            raise ValueError(f"Invalid pooling type: {pooling_type}")

    def build_upsample(self, ratio, upsampling_type):
        return keras.layers.UpSampling2D(size=(ratio, ratio), data_format=self.data_format, interpolation=upsampling_type)


# if __name__ == '__main__':
    # height = 60
    # target_height = 32
    # import math
    # output_shape = math.floor((height - 1) / int((height - 1) // target_height + 1)) + 1
    #
    # print(f"{height=} - {target_height=} -> {output_shape=}")