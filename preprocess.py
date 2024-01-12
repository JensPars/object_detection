import cv2
import json
import selectivesearch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import read_content

# read val_splits.json
with open('val_splits.json') as f:
    val_splits = json.load(f)

train_set = val_splits['train']
example = train_set[0]
#img = 
imgpath, region = read_content(example)
img = cv2.imread(train_set[0].replace(".xml", ".jpg"))
# perform selective search
img, regions = selectivesearch.selective_search(img, scale = 300, sigma = 0.9, min_size = 300)
img = img[:, :, :3]/255.0
# Vizualize the bounding boxes
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(img)
for i in range(len(regions)):
    x1, y1, w, h = regions[i]['rect']
    x2,y2 = x1+w, y1+h
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='r', facecolor='none')
    _ = ax.add_patch(rect)
plt.show()
        



