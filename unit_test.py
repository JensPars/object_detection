import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy

from model import ResNetModule

config = {
    "lr":1e-4,
    "freeze":False,
    "weight_decay":0
}
MODEL = ResNetModule(config=config)

test_image = torch.randint(256, (32, 3, 224, 224)) # to mimic the batches
test_image = test_image.float()


output = MODEL(test_image)
output = output.squeeze()
assert output.shape == torch.Size([32]), f"""Unexpected output size during the forward.
                                            Output size of {output.shape}"""
target = torch.empty(32).random_(2)
loss = MODEL.loss_fn(output, target)

print("Unit tests finished successfully!")