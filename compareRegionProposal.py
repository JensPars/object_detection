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
#⁄ns = len(train_set)
ns  = 2 # number of images to visualize
fig, axss = plt.subplots(ncols=3, nrows=ns, figsize=(10, 5))
vis = True
ss_MABO = []
edge_MABO = []

for n in range(ns):
    example = train_set[n]
    imgpath, regions = read_content(example)
    imgpath = os.path.join(root, imgpath)
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ss_boxes = selective_search_proposal(img)
    edge_boxes = edge_proposal(img, 1000)
    # extract boxes
    ss_iou, ss_idx = [], []
    edge_iou, edge_idx = [], []
    for region in regions:
        ss_io, ss_id = max_iou(region, ss_boxes)
        ss_iou.append(ss_io[0])
        ss_idx.append(ss_id[0])
        edge_io, edge_id = max_iou(region, edge_boxes)
        edge_iou.append(edge_io[0])
        edge_idx.append(edge_id[0])
    ss_MABO.append(np.mean(ss_iou))
    edge_MABO.append(np.mean(edge_iou))
    print(f"Image {n+1}/{ns} done.")

    if vis:
        # Define the color map
        cmap = cm.get_cmap('plasma')
        
        plot_img = img.copy()
        axs = axss[n]
        ax = axs[0]
        for miou,i in zip(edge_iou[:10], edge_idx[:10]):
            x1, y1, x2, y2 = edge_boxes[i]
            plot_img = cv2.rectangle(plot_img, (x1, y1), (x2, y2), 255*np.array(cmap(miou)), 2)
            #ax.text(x2, y2, f"mIoU: {miou:.2f}", color='white', fontsize=8, ha='right', va='bottom')

        ax.set_title('Edge Boxes')
        ax.axis('off')
        plt.tight_layout()
        ax.imshow(plot_img)

        plot_img = img.copy()
        ax = axs[1]
        for miou,i in zip(ss_iou[:10], ss_idx[:10]):
            x1, y1, x2, y2 = ss_boxes[i]
            plot_img = cv2.rectangle(plot_img, (x1, y1), (x2, y2), 255*np.array(cmap(miou)), 2)
            #ax.text(x2, y2, f"mIoU: {miou:.2f}", color='white', fontsize=8, ha='right', va='bottom')

        ax.set_title('Selective Search')
        ax.axis('off')
        plt.tight_layout()
        ax.imshow(plot_img)

        plot_img = img.copy()
        ax = axs[2]
        for x1, y1, x2, y2 in regions:
            plot_img = cv2.rectangle(plot_img, (x1, y1), (x2, y2), (0, 0, 0), 2)

        ax.set_title('Ground Truth')
        ax.axis('off')
        ax.imshow(plot_img)

        # tigh layout
        plt.tight_layout()

        # Create a colorbar legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        # cbar = plt.colorbar(sm, ax=axss, orientation='horizontal')
        # cbar.set_label('mIoU')
        plt.tight_layout()
        plt.savefig("region_proposal.png",bbox_inches='tight', pad_inches=0.2)

        

print(f"Selective Search MABO: {np.mean(ss_MABO):.2f}")
print(f"Edge Boxes MABO: {np.mean(edge_MABO):.2f}")
