import keras
import numpy as np

import keras_cv
from keras_cv.src import layers as cv_layers, bounding_box
from keras_cv.src.backend import ops
# from keras_cv.src.models import RetinaNet
from keras_cv.src.models.object_detection.__internal__ import unpack_input
from keras_cv.src.models.object_detection.efficientdet import efficientdet_presets
from keras_cv.src.models.object_detection.retinanet import RetinaNetLabelEncoder
from keras_cv.src.models.object_detection.retinanet.retinanet import BOX_VARIANCE, _parse_box_loss, \
    _parse_classification_loss, RetinaNet
from keras_cv.src.models.task import Task

class EfficientDet(Task):

    def __init__(self,
                 backbone,
                 extractor_levels,
                 num_classes,
                 bounding_box_format,
                 feature_pyramid,
                 box_head, #todo more defaults
                 classification_head,
                 anchor_generator=None,
                 label_encoder=None,
                 prediction_decoder=None,
                 **kwargs,
                 ):

        if anchor_generator is not None and label_encoder is not None:
            raise ValueError(
                "`anchor_generator` is only to be provided when "
                "`label_encoder` is `None`. Received `anchor_generator="
                f"{anchor_generator}`, label_encoder={label_encoder}`. To "
                "customize the behavior of the anchor_generator inside of a "
                "custom `label_encoder` you should provide both to `RetinaNet`"
                "provide both to `RetinaNet`, and ensure that the "
                "`anchor_generator` provided to both is identical"
            )

        if label_encoder is None:
            anchor_generator = (
                    anchor_generator
                    or self.default_anchor_generator(bounding_box_format)
                    #todo don't import this
            )
            label_encoder = RetinaNetLabelEncoder(
                bounding_box_format=bounding_box_format,
                anchor_generator=anchor_generator,
                box_variance=BOX_VARIANCE,
            )

        outputs = [backbone.get_layer(level).output for level in extractor_levels]
        feature_extractor = keras.Model(inputs=backbone.inputs, outputs=outputs)


        images = keras.layers.Input(
            feature_extractor.input_shape[1:], name="images"
        )
        backbone_outputs = feature_extractor(images)
        feature_pyramid_outputs = feature_pyramid(backbone_outputs)

        cls_preds = classification_head(feature_pyramid_outputs)
        box_preds = box_head(feature_pyramid_outputs)
        cls_pred = []
        box_pred = []
        for cls_layer_pred, box_layer_pred in zip(cls_preds, box_preds):
            box_pred.append(keras.layers.Reshape((-1, 4))(box_layer_pred))
            cls_pred.append(keras.layers.Reshape((-1, num_classes))(cls_layer_pred))

        cls_pred = keras.layers.Concatenate(axis=1, name="classification")(
            cls_pred
        )
        box_pred = keras.layers.Concatenate(axis=1, name="box")(box_pred)
        # box_pred is always in "center_yxhw" delta-encoded no matter what
        # format you pass in.

        inputs = {"images": images}
        outputs = {"box": box_pred, "classification": cls_pred}

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        self.label_encoder = label_encoder
        self.anchor_generator = label_encoder.anchor_generator
        self.bounding_box_format = bounding_box_format
        self.num_classes = num_classes
        self.backbone = backbone

        self.feature_extractor = feature_extractor
        self._prediction_decoder = (
                prediction_decoder
                or cv_layers.NonMaxSuppression(
            bounding_box_format=bounding_box_format,
            from_logits=True,
        )
        )

        self.feature_pyramid = feature_pyramid
        self.classification_head = classification_head
        self.box_head = box_head
        # self.build(backbone.input_shape)

    def compile(
            self,
            box_loss=None,
            classification_loss=None,
            loss=None,
            metrics=None,
            **kwargs,
    ):
        """compiles the EfficientDet.

        compile() mirrors the standard Keras compile() method, but has a few key
        distinctions. Primarily, all metrics must support bounding boxes, and
        two losses must be provided: `box_loss` and `classification_loss`.

        Args:
            box_loss: a Keras loss to use for box offset regression.
                Preconfigured losses are provided when the string "huber" or
                "smoothl1" are passed.
            classification_loss: a Keras loss to use for box classification.
                A preconfigured `FocalLoss` is provided when the string "focal"
                is passed.
            weight_decay: a float for variable weight decay.
            metrics: KerasCV object detection metrics that accept decoded
                bounding boxes as their inputs. Examples of this metric type
                are `keras_cv.metrics.BoxRecall()` and
                `keras_cv.metrics.BoxMeanAveragePrecision()`. When `metrics` are
                included in the call to `compile()`, the RetinaNet will perform
                non-max suppression decoding during the forward pass. By
                default, the RetinaNet uses a
                `keras_cv.layers.MultiClassNonMaxSuppression()` layer to
                perform decoding. This behavior can be customized by passing in
                a `prediction_decoder` to the constructor or by modifying the
                `prediction_decoder` attribute on the model. It should be noted
                that the default non-max suppression operation does not have
                TPU support, and thus when training on TPU metrics must be
                evaluated in a `keras.utils.SidecarEvaluator` or a
                `keras.callbacks.Callback`.
            kwargs: most other `keras.Model.compile()` arguments are supported
                and propagated to the `keras.Model` class.
        """
        if loss is not None:
            raise ValueError(
                "`RetinaNet` does not accept a `loss` to `compile()`. "
                "Instead, please pass `box_loss` and `classification_loss`. "
                "`loss` will be ignored during training."
            )
        box_loss = _parse_box_loss(box_loss)
        classification_loss = _parse_classification_loss(classification_loss)

        if hasattr(classification_loss, "from_logits"):
            if not classification_loss.from_logits:
                raise ValueError(
                    "RetinaNet.compile() expects `from_logits` to be True for "
                    "`classification_loss`. Got "
                    "`classification_loss.from_logits="
                    f"{classification_loss.from_logits}`"
                )
        if hasattr(box_loss, "bounding_box_format"):
            if box_loss.bounding_box_format != self.bounding_box_format:
                raise ValueError(
                    "Wrong `bounding_box_format` passed to `box_loss` in "
                    "`RetinaNet.compile()`. Got "
                    "`box_loss.bounding_box_format="
                    f"{box_loss.bounding_box_format}`, want "
                    "`box_loss.bounding_box_format="
                    f"{self.bounding_box_format}`"
                )

        self.box_loss = box_loss
        self.classification_loss = classification_loss
        losses = {
            "box": self.box_loss,
            "classification": self.classification_loss,
        }
        self._has_user_metrics = metrics is not None and len(metrics) != 0
        self._user_metrics = metrics
        super().compile(loss=losses, **kwargs)


    def compute_loss(self, x, y, y_pred, sample_weight, **kwargs):
        y_for_label_encoder = bounding_box.convert_format(
            y,
            source=self.bounding_box_format,
            target=self.label_encoder.bounding_box_format,
            images=x,
        )
        # print("y_for_label_encoder")
        # for i, x in y_for_label_encoder.items():
        #     print(f"\t{i} {x.shape}")

        boxes, classes = self.label_encoder(x, y_for_label_encoder)

        box_pred = y_pred["box"]
        cls_pred = y_pred["classification"]

        if boxes.shape[-1] != 4:
            raise ValueError(
                "boxes should have shape (None, None, 4). Got "
                f"boxes.shape={tuple(boxes.shape)}"
            )

        if box_pred.shape[-1] != 4:
            raise ValueError(
                "box_pred should have shape (None, None, 4). Got "
                f"box_pred.shape={tuple(box_pred.shape)}. Does your model's "
                "`num_classes` parameter match your losses `num_classes` "
                "parameter?"
            )
        if cls_pred.shape[-1] != self.num_classes:
            raise ValueError(
                "cls_pred should have shape (None, None, 4). Got "
                f"cls_pred.shape={tuple(cls_pred.shape)}. Does your model's "
                "`num_classes` parameter match your losses `num_classes` "
                "parameter?"
            )

        cls_labels = ops.one_hot(
            ops.cast(classes, "int32"), self.num_classes, dtype="float32"
        )
        positive_mask = ops.cast(ops.greater(classes, -1.0), dtype="float32")
        normalizer = ops.sum(positive_mask)
        cls_weights = ops.cast(ops.not_equal(classes, -2.0), dtype="float32")
        cls_weights /= normalizer
        box_weights = positive_mask / normalizer
        y_true = {
            "box": boxes,
            "classification": cls_labels,
        }

        # print(f"{boxes.shape=}")
        # print(f"{cls_labels.shape=}")
        sample_weights = {
            "box": box_weights,
            "classification": cls_weights,
        }
        zero_weight = {
            "box": ops.zeros_like(box_weights),
            "classification": ops.zeros_like(cls_weights),
        }

        sample_weights = ops.cond(
            normalizer == 0,
            lambda: zero_weight,
            lambda: sample_weights,
            )
        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights
        )


    def train_step(self, *args):
        data = args[-1]
        args = args[:-1]
        x, y = unpack_input(data)
        return super().train_step(*args, (x, y))

    def test_step(self, *args):
        data = args[-1]
        args = args[:-1]
        x, y = unpack_input(data)
        return super().test_step(*args, (x, y))

    def compute_metrics(self, x, y, y_pred, sample_weight):
        metrics = {}
        metrics.update(super().compute_metrics(x, {}, {}, sample_weight={}))

        if not self._has_user_metrics:
            return metrics

        y_pred = self.decode_predictions(y_pred, x)

        for metric in self._user_metrics:
            metric.update_state(y, y_pred, sample_weight=sample_weight)

        for metric in self._user_metrics:
            result = metric.result()
            if isinstance(result, dict):
                metrics.update(result)
            else:
                metrics[metric.name] = result
        return metrics

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "label_encoder": keras.saving.serialize_keras_object(
                self.label_encoder
            ),
            "prediction_decoder": self._prediction_decoder,
            "classification_head": keras.saving.serialize_keras_object(
                self.classification_head
            ),
            "box_head": keras.saving.serialize_keras_object(self.box_head),
        }

    @classmethod
    def from_config(cls, config):
        if "box_head" in config and isinstance(config["box_head"], dict):
            config["box_head"] = keras.layers.deserialize(config["box_head"])
        if "classification_head" in config and isinstance(
                config["classification_head"], dict
        ):
            config["classification_head"] = keras.layers.deserialize(
                config["classification_head"]
            )
        if "label_encoder" in config and isinstance(
                config["label_encoder"], dict
        ):
            config["label_encoder"] = keras.layers.deserialize(
                config["label_encoder"]
            )
        if "prediction_decoder" in config and isinstance(
                config["prediction_decoder"], dict
        ):
            config["prediction_decoder"] = keras.layers.deserialize(
                config["prediction_decoder"]
            )
        return super().from_config(config)

    @staticmethod
    def default_anchor_generator(bounding_box_format):
        strides = [2**i for i in range(1, 6)]
        scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        sizes = [256.0, 128.0, 64.0, 32.0, 16.0]
        aspect_ratios = [0.5, 1.0, 2.0]
        return cv_layers.AnchorGenerator(
            bounding_box_format=bounding_box_format,
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
            clip_boxes=True,
        )

if __name__ == '__main__':
    import tensorflow as tf
    images = np.ones((1, 512, 512, 3))
    boxes = tf.cast([[
            [0, 0, 100, 100],
            [100, 100, 200, 200],
            [300, 300, 100, 100],
    ]], dtype=tf.float32)
    labels = {
        "boxes": boxes,
        "classes": tf.cast([[1, 1, 1]], dtype=tf.float32),
     }
    model = efficientdet_presets.from_presets("efficientdet-d0", num_classes=20)

    # print(f"{model.output_shape=}")
    print("CREATED MODEL")
    # Evaluate model without box decoding and NMS
    res = model.predict(images)
    for i, y in res.items():
        print(f"\t{i}: {y.shape}")

    # Prediction with box decoding and NMS
    # model.predict(images)
    #
    # # Train model
    model.compile(
        classification_loss='focal',
        box_loss='smoothl1',
        optimizer=keras.optimizers.SGD(global_clipnorm=10.0),
        jit_compile=True,
    )
    print("DONE COMPILE")
    model.fit(images, labels)

    print(model.count_params())

    print(model.summary())
