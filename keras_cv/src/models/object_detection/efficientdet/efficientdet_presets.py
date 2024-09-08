import keras_cv
from keras_cv.src.models.object_detection.efficientdet.bi_fpn import BiFPN
from keras_cv.src.models.object_detection.efficientdet.detection_heads import ClassNet, BoxNet
from keras_cv.src.models.object_detection.efficientdet.efficientdet import EfficientDet


def from_presets(name:str):
   conf = efficientdet_model_param_dict[name]
   return build_efficientdet(**conf)

def build_efficientdet(
        name: str,
        backbone_name: str,
        image_size: int,
        fpn_num_filters: int,
        fpn_cell_repeats: int,
        box_class_repeats: int,
        fpn_weight_method: str = "fastattn"
):
    num_classes = 90
    backbone = keras_cv.models.EfficientNetV1Backbone.from_preset(
        backbone_name, load_weights=True
    )
    pyramid_level_inputs = sorted(backbone.pyramid_level_inputs.items())
    print(f"{pyramid_level_inputs=}")
    extractor_levels = [level for _, level in pyramid_level_inputs]
    fpn_args = build_fpn_args(fpn_num_filters, fpn_weight_method)
    feature_pyramid = BiFPN(fpn_node_args=fpn_args, depth=5, repeats=fpn_cell_repeats)
    box_head = BoxNet(repeats=box_class_repeats, num_filters=fpn_num_filters)
    class_head = ClassNet(repeats=box_class_repeats, num_filters=fpn_num_filters)
    return EfficientDet(
        backbone=backbone,
        extractor_levels=extractor_levels,
        num_classes=num_classes,
        bounding_box_format="xywh",
        anchor_generator=None,
        label_encoder=None,
        feature_pyramid=feature_pyramid,
        box_head=box_head,
        classification_head=class_head,
    )




# efficientnet_d0 = build_efficientdet(backbone_name="efficientnetv1_b0")


efficientdet_model_param_dict = {
    'efficientdet-d0':
        dict(
            name='efficientdet-d0',
            backbone_name='efficientnetv1_b0',
            image_size=512,
            fpn_num_filters=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
        ),
    'efficientdet-d1':
        dict(
            name='efficientdet-d1',
            backbone_name='efficientnetv1_b1',
            image_size=640,
            fpn_num_filters=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
        ),
    'efficientdet-d2':
        dict(
            name='efficientdet-d2',
            backbone_name='efficientnetv1_b2',
            image_size=768,
            fpn_num_filters=112,
            fpn_cell_repeats=5,
            box_class_repeats=3,
        ),
    'efficientdet-d3':
        dict(
            name='efficientdet-d3',
            backbone_name='efficientnetv1_b3',
            image_size=896,
            fpn_num_filters=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
        ),
    'efficientdet-d4':
        dict(
            name='efficientdet-d4',
            backbone_name='efficientnetv1_b4',
            image_size=1024,
            fpn_num_filters=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    'efficientdet-d5':
        dict(
            name='efficientdet-d5',
            backbone_name='efficientnetv1_b5',
            image_size=1280,
            fpn_num_filters=288,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    'efficientdet-d6':
        dict(
            name='efficientdet-d6',
            backbone_name='efficientnetv1_b6',
            image_size=1280,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            fpn_weight_method='sum',  # Use unweighted sum for stability.
        ),
    'efficientdet-d7':
        dict(
            name='efficientdet-d7',
            backbone_name='efficientnetv1_b6',
            image_size=1536,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=5.0,
            fpn_weight_method='sum',  # Use unweighted sum for stability.
        ),
    'efficientdet-d7x':
        dict(
            name='efficientdet-d7x',
            backbone_name='efficientnetv1_b7',
            image_size=1536,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            # anchor_scale=4.0, #todo scale
            # max_level=8, #todo depth
            fpn_weight_method='sum',  # Use unweighted sum for stability.
        ),
}


def build_fpn_args(num_filters: int, weight_method: str = "fastattn"):
    return {
        "num_filters": num_filters,
        "data_format": "channels_last",
        "weight_method": weight_method,
        "conv_bn_act_pattern": False,
        "separable_conv": True,
        "activation": "swish",
        "apply_batchnorm": True,
        "conv_after_downsample": False,
        "pooling_type": "avg",
        "upsampling_type": "nearest",
    }

if __name__ == '__main__':
    model = from_presets("efficientdet-d0")