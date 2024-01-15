import torch
import os
import sklearn
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from skimage import measure
from skimage.morphology import binary_dilation
import matplotlib.patches as patches

import matplotlib.pyplot as plt
from utils import read_content, max_iou, selective_search_proposal, edge_proposal
from torchvision import transforms as T
from PIL import Image

# DINOv2 without registers
dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')#.to('mps')
# read val_splits.json
with open('val_splits.json') as f:
    val_splits = json.load(f)
root = 'annotated-images'
train_set = val_splits['val']
ns = len(train_set)
vis = False
n = 0
example = train_set[n]
imgpath, gt_regions = read_content(example)
imgpath = os.path.join(root, imgpath)
img = Image.open(imgpath)
orginal_sz = img.size
# resize to multiple of 14
patch_sz = 14
n_patch = 64
img_sz = patch_sz*n_patch
resize = T.Compose([T.Resize(size=(img_sz, img_sz)), T.ToTensor()])
original_img = img.copy()
img = resize(img).unsqueeze(0)

# features
result = dinov2_vits14_reg.forward_features(img)
features = result['x_prenorm'][:,1:,:]

#dbscan = DBSCAN(eps=3, min_samples=2)
#dbscan.fit(features.reshape([-1, 384]).detach().numpy())
#preds = dbscan.labels_.reshape([n_patch, n_patch])

# k-means clustering
kmeans = KMeans(n_clusters=8)
kmeans.fit(features.reshape([-1, 384]).detach().numpy())
preds = kmeans.predict(features.reshape([-1, 384]).detach().numpy())
preds = preds.reshape([n_patch, n_patch])
bboxs = []
for i in np.unique(preds):
    bin_img = (preds == i).astype(int)
    for dil_rate in range(4):
        if dil_rate > 0:
            bin_img = binary_dilation(bin_img)
        # If you want to visualize bounding boxes on the original image
        #fig, ax = plt.subplots()
        #ax.imshow(bin_img.astype('uint8'), cmap='jet')
        #ax.imshow(original_img)
        plot_img = bin_img.copy()#[:,:, np.newaxis]
        #convert to cv2 image
        plot_img = plot_img.astype('uint8')
        plot_img = plot_img*255
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_GRAY2RGB)
        print(plot_img.shape)
        labeled_image = measure.label(bin_img)
        regions = measure.regionprops(labeled_image)
        for region in regions:
            # Draw rectangle around segmented object
            minr, minc, maxr, maxc = region.bbox
            # make int
            minr, minc, maxr, maxc = int(minr), int(minc), int(maxr), int(maxc)

            # map back to original image
            minr = int(minr*orginal_sz[1]/n_patch)
            minc = int(minc*orginal_sz[0]/n_patch)
            maxr = int(maxr*orginal_sz[1]/n_patch)
            maxc = int(maxc*orginal_sz[0]/n_patch)
            plot_img = cv2.rectangle(plot_img, (minc, minr), (maxc, maxr), (255, 0, 0), 1)
            #ax.add_patch(rect)
            bboxs.append([minc, minr, maxc, maxr])
        plt.imshow(plot_img)
        plt.show()
    

plot_img = np.asarray(original_img.copy())
for bbox in bboxs:
    minr, minc, maxr, maxc = bbox
    plot_img = cv2.rectangle(plot_img, (minr, minc), (maxr, maxc), (0, 0, 0), 2)

plt.imshow(plot_img)
plt.show()

plot_img = np.asarray(original_img.copy())
for bbox in gt_regions:
    minr, minc, maxr, maxc = bbox
    plot_img = cv2.rectangle(plot_img, (minr, minc), (maxr, maxc), (0, 0, 0), 2)

plt.imshow(plot_img)
plt.show()


for region in gt_regions:
    print(region)
    minr, minc, maxr, maxc = region
    edge_io, edge_id = max_iou([minr, minc, maxr, maxc], bboxs)
    print(edge_io)


