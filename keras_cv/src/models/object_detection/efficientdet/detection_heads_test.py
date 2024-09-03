import numpy as np

from keras_cv.src.models.object_detection.efficientdet.detection_heads import BoxNet, ClassNet
from keras_cv.src.tests.test_case import TestCase

inputs = [
    np.ones([2, 64, 64, 40]),
    np.ones([2, 32, 32, 40]),
    np.ones([2, 32, 32, 40]),
    np.ones([2, 8, 16, 40]),
    np.ones([2, 8, 8, 40]),
]

class DetectionHeadTest(TestCase):

    def test_box_head(self):
        box_head = BoxNet(num_anchors=3)
        outputs = box_head(inputs)
        for input, output in zip(inputs, outputs):
            self.assertEquals(output.shape, input.shape[:-1] + (3*4,))


    def test_class_head(self):
        class_head = ClassNet(num_anchors=3, num_classes=5)
        outputs = class_head(inputs)
        for input, output in zip(inputs, outputs):
            self.assertEquals(output.shape, input.shape[:-1] + (3*5,))