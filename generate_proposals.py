import cv2
import json
import selectivesearch
import os
import numpy as np
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import read_content, max_iou, selective_search_proposal, edge_proposal
# load the input image


# read val_splits.json
with open('val_splits.json') as f:
    val_splits = json.load(f)
root = 'annotated-images'
train_set = val_splits['val']
ns = len(train_set)
fig, axss = plt.subplots(ncols=3, nrows=ns, figsize=(24, 10*ns))
vis = False
ss_MABO = []
edge_MABO = []

for n in range(ns):
    example = train_set[n]
    imgpath, regions = read_content(example)
    imgpath = os.path.join(root, imgpath)
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edge_boxes = edge_proposal(img)
    # extract boxes
    edge_iou, edge_idx = [], []
    edge_bboxs = []
    for region in regions:
        edge_io, edge_id = max_iou(region, edge_boxes)
        edge_iou.append(edge_io[0])
        edge_idx.append(edge_id[0])
        edge_bboxs.append(edge_boxes[edge_id[0]])
    edge_MABO.append(np.mean(edge_iou))
    print(f"Image {n+1}/{ns} done.")

   

        

print(f"Selective Search MABO: {np.mean(ss_MABO):.2f}")
print(f"Edge Boxes MABO: {np.mean(edge_MABO):.2f}")
