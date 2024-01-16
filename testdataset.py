import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import matplotlib.pyplot as plt

class CustomTestDataset(Dataset):
    def __init__(self, json_file, batch_size=32):
        self.json_file = json_file
        self.data = self._load_data()
        self.keys = list(self.data.keys())
        self.batch_size = batch_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
            T.Resize((224, 224)),
        ])

    def _load_data(self):
        with open(self.json_file) as f:
            data = json.load(f)
        return data

    def _crop_and_resize(self, image, bbox):
        x1, y1, x2, y2 = bbox
        #print(image.shape)
        if x2-x1 > 0 and y2-y1 > 0:
            cropped_image = image[y1:y2, x1:x2]
            #print(cropped_image.shape)
            resized_image = self.transform(cropped_image)  # Adjust the size as needed
            return resized_image
        else:
            #print("skip")
            return
        

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        image_path = self.keys[index]
        image = cv2.imread(image_path)
        _bboxs = self.data[image_path]['proposals']
        bboxs = []
        gt = self.data[image_path]['GT']
        _imgs = []
        for box in _bboxs:
            bboxs.append(torch.Tensor(box['bbox']))
            cropped_image = self._crop_and_resize(image, tuple(box['bbox']))
            if cropped_image is None:
                continue
            _imgs.append(cropped_image)
        # Make sure the output is tensors
        batch = torch.stack(_imgs)
        bboxs = torch.stack(bboxs)
        # Convert GTs to tensors.
        gt = torch.stack([torch.Tensor(g) for g in gt])
        return batch, bboxs, gt
    

# sets = CustomTestDataset("testset.json", batch_size=1)
# example = sets[0]
# print("Shape of batches", example[0].shape)
# print("Shape of bboxs", example[1].shape)
# print("Shape of gt", example[2].shape)
