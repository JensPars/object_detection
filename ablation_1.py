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

#n_regions = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 5000, 10000]
n_regions = np.logspace(1, 4, num=10).astype(int)
n_regions = [1000]
edge_MABO_n = []
ss_MABO_n = []

for n in n_regions:
    edge_MABO = []
    ss_MABO = []
    for i in range(ns):
        example = train_set[i]
        imgpath, regions = read_content(example)
        imgpath = os.path.join(root, imgpath)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edge_boxes = edge_proposal(img, n)
        ss_boxes = selective_search_proposal(img)[:n]
        # extract boxes
        edge_iou, edge_idx = [], []
        ss_iou, ss_idx = [], []
        for region in regions:
            edge_io, edge_id = max_iou(region, edge_boxes)
            edge_iou.append(edge_io[0])
            edge_idx.append(edge_id[0])
            ss_io, ss_id = max_iou(region, ss_boxes)
            ss_iou.append(ss_io)
        print(f"Image {i+1}/{ns} done.")
        #print(f"Edge MABO with {n}: {np.mean(edge_iou):.4f}")
        #print(f"SS MABO with {n}: {np.mean(ss_iou):.4f}")
        #print(edge_MABO)
        edge_MABO.append(np.mean(edge_iou))
        ss_MABO.append(np.mean(ss_iou))
    print(f"Edge MABO with {n}: {np.mean(edge_MABO):.4f}")
    print(f"SS MABO with {n}: {np.mean(ss_MABO):.4f}")
    edge_MABO_n.append(np.mean(edge_MABO))
    ss_MABO_n.append(np.mean(ss_MABO))

# save edge_mabo as npy
plt.plot(edge_MABO)
plt.savefig('plot.png')
np.save('ss_mabo.npy', edge_MABO_n)
np.save('edge_mabo.npy', ss_MABO_n)

   