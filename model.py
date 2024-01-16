# Implementation of the ResNet Model with pretrained weights
# the model is specifically implemented for doing binary classification
# on a pothole dataset

from typing import Any
import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
#from torchmetrics import Accuracy

class ResNetModule(L.LightningModule):

    def __init__(self, config:dict):
        super().__init__()
        self.config = config
        self.model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)

        if self.config['freeze']:
            self._freeze_layers()
        else:
            pass

        self.model.fc = nn.Linear(2048, 1, bias = True) # Defaul init of linear layer

        # Define the loss function 
        self.loss_fn = nn.BCEWithLogitsLoss()
        #self.accuracy = Accuracy(task = "binary")

    def _freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

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
        logits = self(X)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return NotImplementedError()