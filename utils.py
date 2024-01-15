import xml.etree.ElementTree as ET
import numpy as np
import cv2

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
    ious = [bb_intersection_over_union(region, gt_bbox) for gt_bbox in gt_bboxs]
    return -np.sort(-np.array(ious)), np.argsort(-np.array(ious))


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def edge_proposal(img, n):
    '''
    Detects edge proposals in the given image.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        list: A list of bounding boxes representing the detected edge proposals.
            Each bounding box is represented as [x1, y1, x2, y2].
    '''
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
    edges = edge_detection.detectEdges(np.float32(img) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(n)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    # convert boxes to x1, y1, x2, y2
    boxes = [[int(b[0]), int(b[1]), int(b[0]+b[2]), int(b[1]+b[3])] for b in boxes[0]]
    return boxes


def selective_search_proposal(img):
    '''
    Detects region proposals in the given image using selective search.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        list: A list of bounding boxes representing the detected edge proposals.
            Each bounding box is represented as [x1, y1, x2, y2].
    '''
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # selective search
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchQuality()
    boxes = ss.process()
    return boxes