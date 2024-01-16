import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, json_file, k1, k2, batch_size=32):
        self.json_file = json_file
        self.k1 = k1
        self.k2 = k2
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
            print("skip")
            return

    def _calculate_iou(self, bbox1, bbox2):
        # Calculate IoU (Intersection over Union) between two bounding boxes
        # Implement your own IoU calculation logic here
        pass

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        image_path = self.keys[index]
        image = cv2.imread(image_path)
        bboxs = self.data[image_path]['proposals']
        gt = self.data[image_path]['GT']
        IoUs = np.array([val['IoU'] for val in bboxs])
        positive_idx = np.argwhere(IoUs >= self.k2)
        if len(positive_idx) == 0:
            return self.__getitem__(np.random.randint(0, len(self.keys)))
        negative_idx = np.argwhere(IoUs < self.k1)
        # sample 25% positive and 75% negative
        np.random.shuffle(positive_idx)
        np.random.shuffle(negative_idx)
        n_pos = int(self.batch_size*0.25)
        if len(positive_idx) + len(gt) < n_pos:
            n_pos = len(positive_idx) + len(gt)
        n_neg = self.batch_size - n_pos
        pos_idx = positive_idx[:n_pos]
        neg_idx = negative_idx[:n_neg]
        pos_bboxs = [bboxs[int(i)]['bbox'] for i in pos_idx] + gt
        neg_bboxs = [bboxs[int(i)]['bbox'] for i in neg_idx]
        pos_imgs = []
        for bbox in pos_bboxs:
            cropped_image = self._crop_and_resize(image, bbox)
            if cropped_image is None:
                continue
            pos_imgs.append(cropped_image)
        pos_imgs = torch.stack(pos_imgs)
        pos_lbls = torch.ones(pos_imgs.shape[0])
        neg_imgs = []
        for bbox in neg_bboxs:
            cropped_image = self._crop_and_resize(image, bbox)
            neg_imgs.append(cropped_image)
        neg_imgs = torch.stack(neg_imgs)
        neg_lbls = torch.zeros(neg_imgs.shape[0])
        batch = torch.concat((pos_imgs, neg_imgs), dim=0)
        labels = torch.concat((pos_lbls, neg_lbls), dim=0)
        return batch, labels#, pos_bboxs, neg_bboxs, image
    


if __name__ == "__main__":
    json_file = "trainset.json"
    k1 = 0.3
    k2 = 0.7
    batch_size = 32
    dataset = CustomDataset(json_file, k1, k2, batch_size)
    batch, labels, pos_bboxs, neg_bboxs, image = dataset[0]
    vis = True
    if vis:
        # Plot negative and positive boxes on the image using cv2
        for bbox in pos_bboxs:
            x1, y1, x2, y2 = bbox
            # switch x and y
            #x1, y1, x2, y2 = y1, x1, y2, x2
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle for positive boxes
            
        for bbox in neg_bboxs:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle for negative boxes
            
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        for i in range(batch.shape[0]):
            plt.imshow(batch[i].permute(1, 2, 0))
            plt.title(str(labels[i]))
            plt.show()
    
   
