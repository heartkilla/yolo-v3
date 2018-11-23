"""Yolo v3 detection script.

Saves the detections in the `detection` folder.

Usage:
    python detect.py <iou threshold> <confidence threshold> <images>

Example:
    python detect.py 0.5 0.5 data/images/dog.jpg data/images/office.jpg
"""

import tensorflow as tf
import sys

from yolo_v3 import Yolo_v3
from utils import load_images, load_class_names, draw_boxes

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './data/labels/coco.names'
_MAX_OUTPUT_SIZE = 20


def main(iou_threshold, confidence_threshold, img_names):
    batch_size = len(img_names)
    batch = load_images(img_names, model_size=_MODEL_SIZE)
    class_names = load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)

    model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                    max_output_size=_MAX_OUTPUT_SIZE,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)

    inputs = tf.placeholder(tf.float32,
                            [batch_size, *_MODEL_SIZE, 3])

    detections = model(inputs, training=False)

    saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))

    with tf.Session() as sess:
        saver.restore(sess, './weights/model.ckpt')
        detection_result = sess.run(detections, feed_dict={inputs: batch})

    draw_boxes(img_names, detection_result, class_names, _MODEL_SIZE)

    print('Detections has been saved successfully.')


if __name__ == '__main__':
    main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3:])
