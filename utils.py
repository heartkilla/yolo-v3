"""Contains utility functions for Yolo v3 model."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    ims = []

    for img_name in img_names:
        img = Image.open(img_name)
        img_resized = img.resize(size=model_size)
        im = np.array(img_resized, dtype=np.float32)
        im = np.expand_dims(im, axis=0)
        ims.append(im)

    ims = np.concatenate(ims)

    return ims


def load_class_names(file_name):
    """Returns a list of class names read from `file_name`."""
    with open(file_name, 'r') as f:
        class_names = f.readlines()
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
    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names,
                                         boxes_dicts):
        im = Image.open(img_name)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(font='./data/fonts/futur.ttf',
                                  size=(im.size[0] + im.size[1]) // 100)
        resize_factor = \
            (im.size[0] / model_size[0], im.size[1] / model_size[1])
        for cls in range(len(class_names)):
            boxes = boxes_dict[cls]
            if np.size(boxes) != 0:
                color = np.random.permutation([np.random.randint(256), 255, 0])
                for box in boxes:
                    xy, confidence = box[:4], box[4]
                    xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                    x0, y0 = xy[0], xy[1]
                    thickness = (im.size[0] + im.size[1]) // 200
                    for t in np.linspace(0, 1, thickness):
                        xy[0], xy[1] = xy[0] + t, xy[1] + t
                        xy[2], xy[3] = xy[2] - t, xy[3] - t
                        draw.rectangle(xy, outline=tuple(color))
                    text = '{} {:.1f}%'.format(class_names[cls][:-1],
                                               confidence * 100)
                    text_size = draw.textsize(text, font=font)
                    draw.rectangle(
                        [x0, y0 - text_size[1], x0 + text_size[0], y0],
                        fill=tuple(color))
                    draw.text((x0, y0 - text_size[1]), text, fill='black',
                              font=font)

        im.save('./detections/detection_' + str(num + 1) + '.jpg')
