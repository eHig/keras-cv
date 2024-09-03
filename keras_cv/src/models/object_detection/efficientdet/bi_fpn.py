from typing import List

from keras_cv.src.backend import keras

from keras_cv.src.models.object_detection.efficientdet.feature_fusion import FeatureFusion
from keras_cv.src.models.object_detection.efficientdet.op_after_combine import OpAfterCombine
from keras_cv.src.models.object_detection.efficientdet.resample_feature_map import ResampleFeatureMap


class BiFPN(keras.layers.Layer):

    def __init__(self, fpn_node_args, repeats: int, depth: int = None, **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        for i in range(repeats):
            self.blocks.append(BiFPNBlock(fpn_node_args, depth=depth, name=f"fpn_{i}"))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class BiFPNBlock(keras.layers.Layer):

    def __init__(self,
                 fpn_node_args,
                 depth=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fpn_node_args = fpn_node_args
        self.depth = depth
        self.reshapes = []

    # todo ratio as args

    def build(self, input_shapes):
        self.depth = self.depth or len(input_shapes)
        if len(input_shapes) < self.depth:
            #stack layers to input
            for i in range(self.depth - len(input_shapes)):
                self.reshapes.append(
                    ResampleFeatureMap(
                        target_num_channels=self.fpn_node_args["num_filters"],
                        apply_batchnorm=self.fpn_node_args["apply_batchnorm"],
                        conv_after_downsample=self.fpn_node_args["conv_after_downsample"],
                        data_format=self.fpn_node_args["data_format"],
                        pooling_type=self.fpn_node_args["pooling_type"],
                        upsampling_type=self.fpn_node_args["upsampling_type"],
                        name=f"{self.name}-resample-{i}",
                        top_down=False,
                        ratio=2
                    )
                )

        self.top_down_nodes = {
            i: FPNNode(**self.fpn_node_args, name=f"{self.name}_top_down_{i}", top_down=True, ratio=2)
            for i in range(self.depth-1)}
        self.bottom_up_nodes = {
            i: FPNNode(**self.fpn_node_args, name=f"{self.name}_bottom_up_{i}", top_down=False, ratio=2)
            for i in range(1, self.depth)}

    def call(self, inputs, training=False):
        # print(f"{[i.shape for i in inputs]=}")
        for reshape in self.reshapes:
            # print(f"{inputs[-1].shape=}")
            inputs.append(reshape(inputs[-1]))
        n = len(inputs)
        intermediate_outputs = dict()
        intermediate_outputs[n-2] = \
            self.top_down_nodes[n-2]([inputs[n-1], inputs[n-2]])
        for i in range(n-3, 0, -1):
            intermediate_outputs[i] = self.top_down_nodes[i]([intermediate_outputs[i+1], inputs[i]])

        # print(f"{intermediate_outputs.items()=}")

        output = [self.top_down_nodes[0]([intermediate_outputs[1], inputs[0]])]
        for i in range(1, n-1):
            i1 = output[i-1]
            i2 = intermediate_outputs[i]
            i3 = inputs[i]
            output.append(self.bottom_up_nodes[i]([i1, i2, i3]))
        output.append(self.bottom_up_nodes[n-1]([output[n-2], inputs[n-1]]))
        return output


class FPNNode(keras.layers.Layer):

    def __init__(self,
                 num_filters,
                 data_format,
                 apply_batchnorm,
                 conv_after_downsample,
                 pooling_type,
                 upsampling_type,
                 weight_method,
                 conv_bn_act_pattern,
                 separable_conv,
                 activation,
                 top_down: bool,
                 ratio: int,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.ops_after_combine = OpAfterCombine(
            num_filters=num_filters,
            data_format=data_format,
            conv_bn_act_pattern=conv_bn_act_pattern,
            separable_conv=separable_conv,
            activation=activation)
        self.feature_fusion = FeatureFusion(
            data_format=data_format,
            weight_method=weight_method,
            name=f"fuse_{self.name}")
        self.data_format = data_format
        self.num_filters = num_filters
        self.apply_batchnorm = apply_batchnorm
        self.conv_after_downsample = conv_after_downsample
        self.pooling_type = pooling_type
        self.upsampling_type = upsampling_type
        self.top_down = top_down
        self.ratio = ratio

    def build(self, input_shapes):
        self.resample_maps = []
        for i in range(len(input_shapes)):
            self.resample_maps.append(
                ResampleFeatureMap(
                    target_num_channels=self.num_filters,
                    apply_batchnorm=self.apply_batchnorm,
                    conv_after_downsample=self.conv_after_downsample,
                    data_format=self.data_format,
                    pooling_type=self.pooling_type,
                    upsampling_type=self.upsampling_type,
                    downsampling=not self.top_down,
                    ratio=self.ratio if i == 0 else 1,
                    name=f"{self.name}-resample-{i}",
                )
            )

    def call(self, inputs):
        resampled = [func(arg) for func, arg in zip(self.resample_maps, inputs)]
        # print(f"{self.name} - {self.top_down=}")
        # print(f"resampled shapes: {[o.shape for o in resampled]}")
        fused = self.feature_fusion(resampled)
        return self.ops_after_combine(fused)
