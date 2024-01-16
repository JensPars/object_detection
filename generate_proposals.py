import cv2
import json
#import selectivesearch
import os
import numpy as np
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import read_content, max_iou, edge_proposal
# load the input image

import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# # read val_splits.json
# with open('val_splits.json') as f:
#     val_splits = json.load(f)
# root = 'annotated-images'
# train_set = val_splits['val']
# ns = len(train_set)
# n_boxes = 1000
# trainset = {}

# for n in range(ns):
#     example = train_set[n]
#     imgpath, regions = read_content(example)
#     imgpath = os.path.join(root, imgpath)
#     img = cv2.imread(imgpath)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     edge_boxes = edge_proposal(img, n_boxes)
#     # extract boxes
#     proposals = []
#     for region in edge_boxes:
#         edge_io, edge_id = max_iou(region, regions)
#         proposals.append({'bbox': region,
#                            'IoU': edge_io[0]})
#         #print(edge_io[0])
#     trainset[imgpath] = {'proposals': proposals,
#                          'GT': regions}

#     print(f"Image {n+1}/{ns} done.")

   
# # save trainset as json
# with open('valset.json', 'w') as f:
#     json.dump(trainset, f, cls=NpEncoder)

### TEST ###
# read test_splits.json
with open('test_splits.json') as f:
    test_splits = json.load(f)
root = 'annotated-images'
train_set = test_splits['test']
ns = len(train_set)
n_boxes = 1000
testset = {}

for n in range(ns):
    example = train_set[n]
    imgpath, regions = read_content(example)
    # imgpath = os.path.join(root, imgpath)
    # img = cv2.imread(imgpath)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # edge_boxes = edge_proposal(img, n_boxes)
    # extract boxes
    # proposals = []
    # for region in edge_boxes:
        # edge_io, edge_id = max_iou(region, regions)
        # proposals.append({'bbox': region,
        #                    'IoU': edge_io[0]})
        # #print(edge_io[0])
    testset[imgpath] = {
                         'GT': regions}

    print(f"Image {n+1}/{ns} done.")

   
# save trainset as json
with open('testset.json', 'w') as f:
    json.dump(testset, f, cls=NpEncoder)