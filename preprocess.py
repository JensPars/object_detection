import cv2
import json
import selectivesearch

import matplotlib.pyplot as plt
from utils import read_content

# read val_splits.json
with open('val_splits.json') as f:
    val_splits = json.load(f)

train_set = val_splits['train']
img = cv2.imread(train_set[0])


