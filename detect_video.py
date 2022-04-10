from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

center_net_path = './centernet_resnet50_v1_fpn_512x512_coco17_tpu-8/'
pipeline_config = center_net_path + 'pipeline.config'
model_path = center_net_path + 'checkpoint/'
label_map_path = './mscoco_label_map.pbtxt'

cap = cv2.VideoCapture(0)

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_path, 'ckpt-0')).expect_partial()


def get_model_detection_function(model):
    @tf.function
    def detect_fn(image):
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])
    return detect_fn


detect_fn = get_model_detection_function(detection_model)


label_map_path = label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(
    label_map, use_display_name=True)


while 1:
    ret, image_np = cap.read()

    image_np_expanded = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)

    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes']
         [0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    cv2.imshow('object detection', cv2.resize(
        image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()
