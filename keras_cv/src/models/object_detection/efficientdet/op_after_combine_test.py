import numpy as np

from keras_cv.src.models.object_detection.efficientdet.op_after_combine import OpAfterCombine
from keras_cv.src.tests.test_case import TestCase


class OpAfterCombineTest(TestCase):

    def test_op_after_combine_shape(self):
        input = np.ones((1, 32, 32, 16))
        op_after_combine = OpAfterCombine(conv_bn_act_pattern=True,
                                          separable_conv=True,
                                          num_filters=10,
                                          activation="swish",
                                          data_format="channels_last")
        output = op_after_combine(input)
        self.assertEquals(output.shape, input.shape[:-1] + (10,))

