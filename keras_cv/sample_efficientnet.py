import keras
import numpy as np

import keras_cv

if __name__ == '__main__':
    efficientnet = keras_cv.models.EfficientNetV2Backbone.from_preset(
        "efficientnetv2_s"
    )
    image_path ="../imgs/Trafficjamdelhi.jpg"
    image = np.array(keras.utils.load_img(image_path))
    image = keras.ops.image.resize(image, (780, 640))[None, ...]

    outputs = efficientnet.predict(image)
    for i, output in enumerate(outputs):
        print(f"{i}: {output.shape}")