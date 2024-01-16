from model import ResNetModule
from dataset import CustomDataset
from testdataset import CustomTestDataset
from lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import torch

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [torch.cat(data), torch.cat(target)]

train_file = "trainset.json"
val_file = "valset.json"
test_file = "testset.json"
for k1,k2 in [[0.5,0.5]]:#, [0.3, 0.5], [0.3, 0.7], [0.5, 0.7], [0.7, 0.7]]:
    k1 = 0.3
    k2 = 0.7
    batch_size = 32
    dataset = CustomDataset(train_file, k1, k2, batch_size)
    datasetval = CustomDataset(val_file, k1, k2, batch_size)
    testset = CustomTestDataset(test_file, batch_size = 1)
    #batch, labels, pos_bboxs, neg_bboxs, image = dataset[0]
    config = {'lr': 1e-4, 'freeze': True}
    model = ResNetModule(config)
    trainer = Trainer(max_epochs = 1,
                      precision = '16-mixed')
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle = True, collate_fn=my_collate)
    val_dataloader = DataLoader(datasetval, batch_size=1, shuffle = False, collate_fn=my_collate)
    test_dataloader = DataLoader(testset, batch_size = 1, shuffle = False)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)