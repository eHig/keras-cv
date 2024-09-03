import numpy as np

from keras_cv.src.models.object_detection.efficientdet.feature_fusion import FeatureFusion
from keras_cv.src.tests.test_case import TestCase


class FeatureFusionTest(TestCase):

    def test_attn_fusion(self):
        h = 32
        w = 32
        channels = 8
        ones = np.ones((1, h, w, channels))
        twos = np.ones((1, h, w, channels)) * 2
        fives = np.ones((1, h, w, channels)) * 5
        two_inputs = (ones, twos)
        three_inputs = (ones, twos, fives)
        for method in ["attn", "channel_attn"]:
            fusion = FeatureFusion(weight_method=method)
            result = fusion(inputs=two_inputs)
            expected = np.ones((1, h, w, channels)) * 1.5
            self.assertAllClose(expected, result)
            fusion = FeatureFusion(weight_method=method)
            result = fusion(inputs=three_inputs)
            expected = np.ones((1, h, w, channels)) * 8/3
            self.assertAllClose(expected, result)

    def test_fastattn_fusion(self):
        h = 32
        w = 32
        channels = 8
        ones = np.ones((1, h, w, channels))
        twos = np.ones((1, h, w, channels)) * 2
        fives = np.ones((1, h, w, channels)) * 5
        two_inputs = (ones, twos)
        three_inputs = (ones, twos, fives)
        for method in ["fastattn", "channel_fastattn"]:
            fusion = FeatureFusion(weight_method=method)
            result = fusion(inputs=two_inputs)
            expected = np.ones((1, h, w, channels)) * (3/2.0001)
            self.assertAllClose(expected, result)
            fusion = FeatureFusion(weight_method=method)
            result = fusion(inputs=three_inputs)
            expected = np.ones((1, h, w, channels)) * (8/3.0001)
            self.assertAllClose(expected, result)
