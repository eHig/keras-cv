import keras_cv
import numpy as np
import tensorflow as tf


if __name__ == '__main__':

    images = np.ones((1, 512, 512, 3))
    labels = {
        "boxes": tf.cast([
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 100, 100],
            ]
        ], dtype=tf.float32),
        "classes": tf.cast([[1, 1, 1]], dtype=tf.float32),
    }
    model = keras_cv.models.RetinaNet(
        num_classes=20,
        bounding_box_format="xywh",
        backbone=keras_cv.models.ResNet50Backbone.from_preset(
            "resnet50_imagenet"
        )
    )

# Evaluate model without box decoding and NMS
model(images)

# Prediction with box decoding and NMS
model.predict(images)