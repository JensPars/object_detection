import cv2
import json
import selectivesearch
import os
import numpy as np
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import read_content, max_iou, selective_search_proposal, edge_proposal
from glob import glob

# load the input image
paths = glob('/Users/jensparslov/Downloads/archive/VOC2012_test/VOC2012_test/Annotations/*.xml')
# sample 100 images by shuffle
np.random.shuffle(paths)
paths = paths[:100]

root = '/Users/jensparslov/Downloads/archive/VOC2012_test/VOC2012_test/JPEGImages'
train_set = paths
ns = len(train_set)
edge_MABO = []
for n in [1000]:
    for i in range(ns):
        example = train_set[i]
        imgpath, regions = read_content(example)
        imgpath = os.path.join(root, imgpath)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edge_boxes = edge_proposal(img, n)
        # extract boxes
        edge_iou, edge_idx = [], []
        for region in regions:
            edge_io, edge_id = max_iou(region, edge_boxes)
            edge_iou.append(edge_io[0])
            edge_idx.append(edge_id[0])
        edge_MABO.append(np.mean(edge_iou))
        print(f"Image {i+1}/{ns} done.")

    print(f"Edge MABO with {n}: {np.mean(edge_MABO):.4f}")
   