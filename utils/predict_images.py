from posixpath import split
import cv2
import glob
import sys
import numpy as np
import tensorflow as tf
from random import shuffle
import time
import json
import colorsys
import sys
sys.path.append('../src/')
from model import create_model, yolo_eval
from utils import get_classes, get_anchors, letterbox_image

'''
------------------------------------------------------- 
Predicts on a random image from annotation file or data folder
------------------------------------------------------- 
'''

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

img_dir_path = sys.argv[1]
classes_path = sys.argv[2]
anchors_path = sys.argv[3]
weights_path = sys.argv[4]

# If we're loading in an annotation file, collect image paths
if img_dir_path.split(".")[-1] == 'txt':
    with open(img_dir_path, "r") as f:
        lines = f.readlines()
    img_paths = [l.split(" ")[0] for l in lines]
else:
    img_paths = glob.glob(img_dir_path + "/*")

# -----------------------------------------------------------------

def show_image(img, img_path):
    cv2.imshow(img_path,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
input_shape = (416,416)

# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
                for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
np.random.seed(10101)  # Fixed seed for consistent colors across runs.
np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
np.random.seed(None)  # Reset seed to default.

model = create_model(input_shape, anchors, num_classes, weights_path=weights_path, inference_only=True)

max_boxes = 50
score_thresh = 0.10
iou_thresh = 0.90

shuffle(img_paths)

for img_path in img_paths:
    img = cv2.imread(img_path)

    image_data = np.divide(img, 255., casting="unsafe")
    image_data = np.expand_dims(image_data, 0)

    start = time.time()
    outputs = model.predict(image_data, verbose=0)
    end = time.time()
    out_boxes, out_scores, out_classes = yolo_eval(outputs, anchors, num_classes, input_shape, max_boxes, score_thresh, iou_thresh)
                
    print('Detected {} objects in {}s ({})'.format(len(out_boxes), (end-start), img_path))

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(img.shape[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(img.shape[0], np.floor(right + 0.5).astype('int32'))

        print(label, (left, top), (right, bottom))
        cv2.rectangle(img, (left, top), (right, bottom), (255,0,0), 1)

    show_image(img, "predicted detections")
    