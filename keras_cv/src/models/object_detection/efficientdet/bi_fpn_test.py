import numpy as np

from keras_cv.src.models.object_detection.efficientdet.bi_fpn import FPNNode, BiFPNBlock, BiFPN
from keras_cv.src.tests.test_case import TestCase


default_args = {
    "num_filters": 25,
    "data_format":"channels_last",
    "weight_method":"channel_fastattn",
    "conv_bn_act_pattern":True,
    "separable_conv": False,
    "activation":"swish",
    "apply_batchnorm":True,
    "conv_after_downsample": True,
    "pooling_type": "avg",
    "upsampling_type": "nearest",
}

inputs = [
    np.ones([2, 64, 64, 3]),
    np.ones([2, 32, 32, 10]),
    np.ones([2, 16, 16, 30]),
    np.ones([2, 8, 8, 50]),
]


class BiFpnTest(TestCase):

    def test_fnode_shape(self):

        for i in range(0, len(inputs)-1):
            large = inputs[i]
            small = inputs[i+1]
            downsample = FPNNode(ratio=2, top_down=True, **default_args)
            features = [large, small]
            if i%2 == 0:
                features.append(small)
            downsampled = downsample(features)
            self.assertEquals(downsampled.shape, small.shape[:-1] + (25,))
            upsample = FPNNode(ratio=2, top_down=False, **default_args)
            features = [small, large]
            if i%2 == 0:
                features.append(large)
            upsampled = upsample(features)
            self.assertEquals(upsampled.shape, large.shape[:-1] + (25,))

    def test_bi_fpn_block(self):
        block = BiFPNBlock(fpn_node_args=default_args)
        outputs = block(inputs)
        for i in range(len(inputs)):
            self.assertEquals(outputs[i].shape, inputs[i].shape[:-1] + (25,))

    def test_bi_fpn_block_with_depth(self):
        block = BiFPNBlock(fpn_node_args=default_args, depth=7)
        outputs = block(inputs)
        for i in range(len(inputs)):
            self.assertEquals(outputs[i].shape, inputs[i].shape[:-1] + (25,))
        self.assertEquals(outputs[-2].shape, (2, 4, 4, 25))
        self.assertEquals(outputs[-1].shape, (2, 2, 2, 25))

    def test_bi_fpn(self):
        block = BiFPN(fpn_node_args=default_args, repeats=2)
        outputs = block(inputs)
        for i in range(len(inputs)):
            self.assertEquals(outputs[i].shape, inputs[i].shape[:-1] + (25,))

    def test_bi_fpn_with_depth(self):
        block = BiFPN(fpn_node_args=default_args, repeats=2, depth=7)
        outputs = block(inputs)
        for i in range(len(inputs)):
            self.assertEquals(outputs[i].shape, inputs[i].shape[:-1] + (25,))
        self.assertEquals(outputs[-2].shape, (2, 4, 4, 25))
        self.assertEquals(outputs[-1].shape, (2, 2, 2, 25))
