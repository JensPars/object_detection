import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from utils import bb_intersection_over_union

def mAP(bboxes, scores, ground_truth, threshold = 0.5):
    filtered_boxes_idx = torchvision.ops.nms(bboxes, scores, threshold)
    bboxes_kept = bboxes[filtered_boxes_idx]
    scores_kept = scores[filtered_boxes_idx]
    scores_sorted, order = torch.sort(scores_kept, descending = True)
    bboxes_sorted = bboxes_kept[order] 

    n_ground_truth = len(ground_truth)
    T = 0
    F = 0
    Recall = [0]
    Precision = [1]
    
    for box in bboxes_sorted:
        if len(ground_truth) == 0:
          break

        iou_list = []
    
        if n_ground_truth == 1:
            iou_list.append(bb_intersection_over_union(box.cpu().numpy(), gt.cpu().numpy()))
        else:
            for gt in [ground_truth]:
                print("Ground Truths: ", gt)
                print("Box: ", box)
                iou_list.append(bb_intersection_over_union(box.cpu().numpy(), gt.cpu().numpy()))
        iou_list = np.array(iou_list)
        if np.max(iou_list) > 0.5:
            idx = np.argmax(iou_list)
            ground_truth = np.delete(ground_truth, idx, 0)
            T += 1
        else:
            F += 1
        
        Precision.append(T / (T+F))
        Recall.append(T / n_ground_truth)
        
    Area = np.trapz(Precision, Recall)
    return Recall, Precision, Area

def plot_mAP(Recall, Precision, Area):
    formatted_area = "{:.3f}".format(Area)

    # Create a plot
    plt.plot(Recall, Precision)
    plt.fill_between(Recall, Precision, color='skyblue', alpha=0.4, label='Area under Curve')


    # Add axis titles
    plt.xlabel('Recall')
    plt.ylabel('Precision')


    # Add a title to the plot
    plt.title(f'Approximate area under the curve: {formatted_area}')

    # Show the plot
    plt.show()
