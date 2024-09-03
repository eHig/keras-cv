import numpy as np

from keras_cv.src.models.object_detection.efficientdet.resample_feature_map import ResampleFeatureMap
from keras_cv.src.tests.test_case import TestCase

class ResampleFeatureMapTest(TestCase):

    def test_resample_output_shape(self):
        inputs = [
            np.ones([2, 64, 64, 3]),
            np.ones([2, 32, 32, 10]),
            np.ones([2, 16, 16, 30]),
            np.ones([2, 8, 8, 50]),
        ]
        for i in range(0, len(inputs)-1):
            large = inputs[i]
            small = inputs[i+1]
            downsample = ResampleFeatureMap(25, pooling_type="avg", ratio=2, downsampling=True)
            downsampled = downsample(large)
            self.assertEquals(downsampled.shape[:-1], small.shape[:-1])
            self.assertEquals(downsampled.shape[-1], 25)
            upsample = ResampleFeatureMap(25, ratio=2, downsampling=False, upsampling_type="nearest")
            upsampled = upsample(small)
            self.assertEquals(upsampled.shape[:-1], large.shape[:-1])
            self.assertEquals(upsampled.shape[-1], 25)
