import cv2
import json
import os
import selectivesearch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import read_content, max_iou

# read val_splits.json
with open('val_splits.json') as f:
    val_splits = json.load(f)
root = 'annotated-images'
train_set = val_splits['train']
example = train_set[0]
#img = 
imgpath, gt_regions = read_content(example)
# join path
imgpath = os.path.join(root, imgpath)
img = cv2.imread(imgpath)
#img = img/255.0
# resize image
img_sz = 720
orginal_sz = 720
if orginal_sz != img_sz:
    img = cv2.resize(img, (img_sz, img_sz))

# perform selective search
img, prop_regions = selectivesearch.selective_search(img, scale = 200, sigma = 0.9, min_size = 300)
prop_regions = [region['rect'] for region in prop_regions]

abo, idx = max_iou(gt_regions[0], prop_regions)
print(abo)







