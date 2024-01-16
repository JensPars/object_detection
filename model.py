# Implementation of the ResNet Model with pretrained weights
# the model is specifically implemented for doing binary classification
# on a pothole dataset

from typing import Any
import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics import Accuracy
from torchvision import transforms as T
from utils import edge_proposal
from mAP import mAP

class ResNetModule(L.LightningModule):

    def __init__(self, config:dict):
        super().__init__()
        self.config = config
        self.model = resnet18(weights = ResNet18_Weights.DEFAULT)

        if self.config['freeze']:
            self._freeze_layers()
        else:
            pass

        self.model.fc = nn.Linear(512, 1, bias = True) # Defaul init of linear layer

        # Define the loss function 
        self.loss_fn = nn.BCEWithLogitsLoss()
        #self.accuracy = Accuracy(task = "binary")

        # Transform for crops:
        self.transform = T.Compose([
            T.ToTensor(),
            #T.Normalize(mean=[0.485, 0.456, 0.406], 
            #            std=[0.229, 0.224, 0.225]),
            T.Resize((224, 224)),
        ])

    def _freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _crop_and_resize(self, image, bbox):
        x1, y1, x2, y2 = bbox
        cropped_image = image[x1:x2, y1:y2]
        resized_image = self.transform(cropped_image)  # Adjust the size as needed
        return resized_image
    
    def _genRegionProps(self, X, method:str="edge"):
        if method == "edge":
            assert X.shape[0] == 3, "The channels batch dimension is not removed."
            bboxs = edge_proposal(X, 1000) # TODO: Might an argmument we would want in config.
            imgs = [self._crop_and_resize(X, bbox) for bbox in bboxs] 
            assert len(bboxs) == imgs, "Different lengths of the output."       
        elif method == "ss":
            raise NotImplementedError("Method is not yet supported. Try 'edge'.")
        else:
            raise ValueError("Unknown method for generating region proposals. Try 'ss' or 'edge'")
        return imgs, bboxs

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # X is a n x n Tensor and y is a single value specifying
        # whether the image is a pothole or not
        X, y = batch 
        logits = self(X).squeeze()
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X).squeeze()
        loss = self.loss_fn(logits, y)
        #acc = self.accuracy(F.sigmoid(logits), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # TODO: Implement the logic need for assesing the performance on the test set.
        X, y = batch
        preds = []
        for i in range(X.shape[0]):
            _img = X[i,:,:,:].squeeze().numpy().cpu()
            crops, bboxes = self._genRegionProps(_img)
            for crop in crops:
                crop = crop.to("cuda")
                preds.append(F.sigmoid(self(crop)))
        # Convert to tensors
        preds = torch.stack(preds, dim = 0)
        bboxes = torch.Tensor(bboxes)
        rec, pre, ar = mAP(bboxes=bboxes, scores = preds, ground_truth=y, threshold=0.5)
        self.log("mAP", ar, on_step=False, on_epoch=True, prog_bar=True, logger=True)