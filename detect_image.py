import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np
import tensorflow as tf
import pathlib
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import warnings

matplotlib.use('TkAgg')

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:

    tf.config.experimental.set_memory_growth(gpu, True)

PATH_TO_LABELS = "mscoco_label_map.pbtxt"

PATH_TO_SAVED_MODEL = "centernet_resnet50_v1_fpn_512x512_coco17_tpu-8\saved_model"

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

image_np = np.array(Image.open(os.sys.argv[1]))

input_tensor = tf.convert_to_tensor(image_np)

input_tensor = input_tensor[tf.newaxis, ...]

detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))

detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

detections['detection_classes'] = detections['detection_classes'].astype(
    np.int64)

image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.30,
    agnostic_mode=False)

plt.figure()

plt.imshow(image_np_with_detections)

plt.show()

if cv2.waitKey(25) & 0xFF == ord('q'):

    exit()
