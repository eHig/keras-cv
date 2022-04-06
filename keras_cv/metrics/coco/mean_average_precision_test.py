# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for COCOMeanAveragePrecision."""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv.metrics import COCOMeanAveragePrecision


class COCOMeanAveragePrecisionTest(tf.test.TestCase):
    def test_runs_inside_model(self):
        i = keras.layers.Input((None, None, 6))
        model = keras.Model(i, i)

        mean_average_precision = COCOMeanAveragePrecision(
            max_detections=100,
            num_buckets=4,
            class_ids=[1],
            area_range=(0, 64**2),
        )

        # These would match if they were in the area range
        y_true = np.array([[[0, 0, 10, 10, 1], [5, 5, 10, 10, 1]]]).astype(np.float32)
        y_pred = np.array([[[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.5]]]).astype(
            np.float32
        )

        model.compile(metrics=[mean_average_precision])

        # mean_average_precision.update_state(y_true, y_pred)

        model.evaluate(y_pred, y_true)

        self.assertAllEqual(mean_average_precision.result(), 1.0)

    def test_first_buckets_have_no_boxes(self):
        mean_average_precision = COCOMeanAveragePrecision(
            iou_thresholds=[0.33],
            class_ids=[1],
            max_detections=100,
            num_buckets=4,
            recall_thresholds=[0.3, 0.5],
        )

        ground_truths = [3]
        # one class
        true_positives = [
            [
                [
                    # one threshold
                    # three buckets
                    0,
                    0,
                    1,
                    2,
                ]
            ]
        ]
        false_positives = [
            [
                [
                    # one threshold
                    # three buckets
                    0,
                    0,
                    1,
                    0,
                ]
            ]
        ]

        # so we get:
        # rcs = [0, 0, 0.33,  1.0]
        # prs = [NaN, NaN, 0.5 ,  0.75]
        # after filtering:
        # rcs = [0.33, 1.0]
        # prs = [0.5, 0.75]
        # so for PR pairs we get:
        # [0.3, 0.5]
        # [0.5, 0.75]

        # So mean average precision should be: (0.5 + 0.75)/2 = 0.625.
        ground_truths = tf.constant(ground_truths, tf.int32)
        true_positives = tf.constant(true_positives, tf.int32)
        false_positives = tf.constant(false_positives, tf.int32)

        mean_average_precision.ground_truths.assign(ground_truths)
        mean_average_precision.true_positive_buckets.assign(true_positives)
        mean_average_precision.false_positive_buckets.assign(false_positives)

        self.assertEqual(mean_average_precision.result(), 0.625)

    def test_result_method_with_direct_assignment_one_threshold(self):
        mean_average_precision = COCOMeanAveragePrecision(
            iou_thresholds=[0.33],
            class_ids=[1],
            max_detections=100,
            num_buckets=3,
            recall_thresholds=[0.3, 0.5],
        )

        ground_truths = [3]

        # one class
        true_positives = [
            [
                [
                    # one threshold
                    # three buckets
                    0,
                    1,
                    2,
                ]
            ]
        ]

        false_positives = [
            [
                [
                    # one threshold
                    # three buckets
                    1,
                    0,
                    0,
                ]
            ]
        ]

        # so we get:
        # rcs = [0, 0.33,  1.0]
        # prs = [0, 0.5 ,  0.75]

        # so for PR pairs we get:
        # [0.3, 0.5]
        # [0.5, 0.75]

        # So mean average precision should be: (0.5 + 0.75)/2 = 0.625.

        ground_truths = tf.constant(ground_truths, tf.int32)
        true_positives = tf.constant(true_positives, tf.int32)
        false_positives = tf.constant(false_positives, tf.int32)

        mean_average_precision.ground_truths.assign(ground_truths)
        mean_average_precision.true_positive_buckets.assign(true_positives)
        mean_average_precision.false_positive_buckets.assign(false_positives)

        self.assertEqual(mean_average_precision.result(), 0.625)

    def test_result_method_with_direct_assignment_missing_class(self):
        mean_average_precision = COCOMeanAveragePrecision(
            iou_thresholds=[0.33],
            class_ids=[1, 2],
            max_detections=100,
            num_buckets=3,
            recall_thresholds=[0.3, 0.5],
        )

        ground_truths = [3, 0]

        # one class
        true_positives = [
            [
                [
                    0,
                    1,
                    2,
                ]
            ],
            [
                [
                    0,
                    0,
                    0,
                ]
            ],
        ]

        false_positives = [
            [[1, 0, 0]],
            [[0, 0, 0]],
        ]
        # Result should be the same as above.
        ground_truths = tf.constant(ground_truths, tf.int32)
        true_positives = tf.constant(true_positives, tf.int32)
        false_positives = tf.constant(false_positives, tf.int32)

        mean_average_precision.ground_truths.assign(ground_truths)
        mean_average_precision.true_positive_buckets.assign(true_positives)
        mean_average_precision.false_positive_buckets.assign(false_positives)

        self.assertEqual(mean_average_precision.result(), 0.625)