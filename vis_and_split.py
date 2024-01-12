import cv2
import json

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

# read splits.json
with open('splits.json') as f:
    splits = json.load(f)
train = splits['train']
from pathlib import Path
root = Path('annotated-images')

paths = [str(root/path) for path in train]
# sample 80/20 train/val split
np.random.seed(0)
np.random.shuffle(paths)
train_splits = int(0.8 * len(paths))
val_splits = {}
val_splits["train"] = paths[:train_splits]
val_splits["val"] = paths[train_splits:]
# save as json
with open('val_splits.json', 'w') as f:
    json.dump(val_splits, f)

# read annotations
for path in paths:
    name, boxes = read_content(path)
    # plot images with bboxes
    img = cv2.imread(str(root/name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # plot bbox
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    break
