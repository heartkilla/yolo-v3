# Yolo v3 Object Detection in Tensorflow
Yolo v3 is an algorithm using deep convolutional neural networks to detect objects.

## Getting started

### Prerequisites
This project is written in Python 3.6.6 using Tensorflow (deep learning), NumPy (numerical computing) and Pillow (image processing) packages.

```
pip install -r requirements
```

### Downloading official pretrained weights
Let's download official weights pretrained on COCO dataset. 

```
wget -P weights https://pjreddie.com/media/files/yolov3.weights
```

### Save the weights in Tensorflow format
Save the weights using `load_weights.py` script.

```
python load_weights.py
```

## Running the model
Now you can run the model using `detect.py` script. Don't forget to setup the IoU (Interception over Union) and confidence thresholds.
### Usage
```
python detect.py <iou threshold> <confidence threshold> <images>
```
### Example
Let's run an example using official sample images.
```
python detect.py 0.5 0.5 data/images/dog.jpg data/images/office.jpg
```
Then you can find the detections in the `detection` folder.
