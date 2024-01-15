import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, json_file, iou_threshold):
        self.json_file = json_file
        self.iou_threshold = iou_threshold
        self.data = self._load_data()

    def _load_data(self):
        with open(self.json_file) as f:
            data = json.load(f)
        return data

    def _crop_and_resize(self, image, bbox):
        x, y, w, h = bbox
        cropped_image = image[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, (224, 224))  # Adjust the size as needed
        return resized_image

    def _calculate_iou(self, bbox1, bbox2):
        # Calculate IoU (Intersection over Union) between two bounding boxes
        # Implement your own IoU calculation logic here
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image_path = item['image_path']
        bbox = item['bbox']
        class_label = item['class_label']

        image = cv2.imread(image_path)
        cropped_image = self._crop_and_resize(image, bbox)

        # Calculate IoU between the cropped image's bounding box and the original bounding box
        iou = self._calculate_iou(bbox, cropped_image_bbox)

        # Conditionally assign class label based on the IoU threshold
        if iou >= self.iou_threshold:
            return torch.from_numpy(cropped_image), class_label
        else:
            return None, None
