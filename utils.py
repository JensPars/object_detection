import xml.etree.ElementTree as ET
import numpy as np

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

def check_proposal(region, gt_bboxs, threshold:tuple=(0.5, 0.5)):
    iou = max_iou(region, gt_bboxs)
    if isinstance(threshold, float):
        k1 = k2 = threshold
    elif isinstance(threshold, tuple):
        k1, k2 = threshold
    
    if iou < k1:
        return False # Background
    elif iou >= k1 and iou < k2:
        return None # Ignore
    elif iou > k2:
        return True
    

def max_iou(region, gt_bboxs):
    ious = [_iou(region, gt_bbox) for gt_bbox in gt_bboxs]
    return max(ious), np.argmax(ious)


def _iou(region, gt_bbox):
    x1, y1, h, w = region
    x2, y2 = x1+h, y1+w
    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
    # compute intersection
    x1_i = max(x1, x1_gt)
    y1_i = max(y1, y1_gt)
    x2_i = min(x2, x2_gt)
    y2_i = min(y2, y2_gt)
    intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    # compute union
    area_region = (x2 - x1) * (y2 - y1)
    area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union = area_region + area_gt - intersection
    # compute iou
    iou = intersection / union
    return iou