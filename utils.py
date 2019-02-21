"""Contains utility functions for Yolo v3 model."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from seaborn import color_palette
import cv2


def load_images(img_names, model_size):
    """Loads images in a 4D array.
    Args:
        img_names: A list of images names.
        model_size: The input size of the model.
        data_format: A format for the array returned
            ('channels_first' or 'channels_last').
    Returns:
        A 4D NumPy array.
    """
    imgs = []

    for img_name in img_names:
        img = Image.open(img_name)
        img = img.resize(size=model_size)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img[:, :, :3], axis=0)
        imgs.append(img)

    imgs = np.concatenate(imgs)

    return imgs


def load_class_names(file_name):
    """Returns a list of class names read from `file_name`."""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    """Draws detected boxes.
    Args:
        img_names: A list of input images names.
        boxes_dict: A class-to-boxes dictionary.
        class_names: A class names list.
        model_size: The input size of the model.
    Returns:
        None.
    """
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names,
                                         boxes_dicts):
        img = Image.open(img_name)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font='./data/fonts/futur.ttf',
                                  size=(img.size[0] + img.size[1]) // 100)
        resize_factor = \
            (img.size[0] / model_size[0], img.size[1] / model_size[1])
        for cls in range(len(class_names)):
            boxes = boxes_dict[cls]
            if np.size(boxes) != 0:
                color = colors[cls]
                for box in boxes:
                    xy, confidence = box[:4], box[4]
                    xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                    x0, y0 = xy[0], xy[1]
                    thickness = (img.size[0] + img.size[1]) // 200
                    for t in np.linspace(0, 1, thickness):
                        xy[0], xy[1] = xy[0] + t, xy[1] + t
                        xy[2], xy[3] = xy[2] - t, xy[3] - t
                        draw.rectangle(xy, outline=tuple(color))
                    text = '{} {:.1f}%'.format(class_names[cls],
                                               confidence * 100)
                    text_size = draw.textsize(text, font=font)
                    draw.rectangle(
                        [x0, y0 - text_size[1], x0 + text_size[0], y0],
                        fill=tuple(color))
                    draw.text((x0, y0 - text_size[1]), text, fill='black',
                              font=font)
                    print('{} {:.2f}%'.format(class_names[cls],
                                              confidence * 100))

        rgb_img = img.convert('RGB')

        rgb_img.save('./detections/detection_' + str(num + 1) + '.jpg')


def draw_frame(frame, frame_size, boxes_dicts, class_names, model_size):
    """Draws detected boxes in a video frame.
    Args:
        frame: A video frame.
        frame_size: A tuple of (frame width, frame height).
        boxes_dicts: A class-to-boxes dictionary.
        class_names: A class names list.
        model_size:The input size of the model.
    Returns:
        None.
    """
    boxes_dict = boxes_dicts[0]
    resize_factor = (frame_size[0] / model_size[1], frame_size[1] / model_size[0])
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for cls in range(len(class_names)):
        boxes = boxes_dict[cls]
        color = colors[cls]
        color = tuple([int(x) for x in color])
        if np.size(boxes) != 0:
            for box in boxes:
                xy = box[:4]
                xy = [int(xy[i] * resize_factor[i % 2]) for i in range(4)]
                cv2.rectangle(frame, (xy[0], xy[1]), (xy[2], xy[3]), color[::-1], 2)
                (test_width, text_height), baseline = cv2.getTextSize(class_names[cls],
                                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                                      0.75, 1)
                cv2.rectangle(frame, (xy[0], xy[1]),
                              (xy[0] + test_width, xy[1] - text_height - baseline),
                              color[::-1], thickness=cv2.FILLED)
                cv2.putText(frame, class_names[cls], (xy[0], xy[1] - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
