import cv2
import sys
from random import shuffle
import numpy as np

def show_image(img, img_path):
    cv2.imshow(img_path,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

annotation_file = sys.argv[1]

with open(annotation_file, 'r') as f:
    lines = f.readlines()

shuffle(lines)

for line in lines:
    print(line)
    data = line.split(" ")
    img = cv2.imread(data[0])
    for box_line in data[1:]:
        box_line = box_line.rstrip('\n')
        box_coords = box_line.split(",")
        if box_coords[0] != '':
            cv2.rectangle(img, (int(box_coords[0]), int(box_coords[1])), (int(box_coords[2]), int(box_coords[3])), (255,0,0), 1)
    show_image(img, "t")
