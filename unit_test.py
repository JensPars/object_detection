import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy

from model import ResNetModule
from mAP import mAP
from testdataset import CustomTestDataset

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


# Two propals are the exact box.
proposal = torch.Tensor([[86, 63, 131, 82],
                        [358, 10, 400, 61],
                        [106, 10, 221, 119]])

scores = torch.Tensor([0.9, 0.8, 0.99])

gt = torch.Tensor([[358, 10, 400, 61],
                    [106, 10, 221, 119]])
print(mAP(bboxes=proposal, scores=scores, ground_truth=gt))
print("Unit tests finished successfully!")


