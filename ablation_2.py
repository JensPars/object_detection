from model import ResNetModule
from dataset import CustomDataset
from testdataset import CustomTestDataset
from lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [torch.cat(data), torch.cat(target)]

train_file = "trainset.json"
val_file = "valset.json"
test_file = "testsetv2.json"
i = 6
for k1,k2 in [[0.5,0.5]]:#, [0.3, 0.5], [0.3, 0.7], [0.5, 0.7], [0.7, 0.7]]:
    k1, k2 = (0.1, 0.9)
    model_name = f"model{i}"
    batch_size = 32
    dataset = CustomDataset(train_file, k1, k2, batch_size)
    datasetval = CustomDataset(val_file, k1, k2, batch_size)
    testset = CustomTestDataset(test_file, batch_size = 1)
    #batch, labels, pos_bboxs, neg_bboxs, image = dataset[0]
    config = {'lr': 1e-4, 'freeze': False}
    model = ResNetModule(config)
    checkpoint_callback = ModelCheckpoint(monitor = "val_acc", dirpath = f"models/{model_name}", mode = "max")
    trainer = Trainer(max_epochs = 20,
                      callbacks = [checkpoint_callback],
                      precision = '16-mixed')
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle = True, collate_fn=my_collate, num_workers = 5)
    val_dataloader = DataLoader(datasetval, batch_size=1, shuffle = False, collate_fn=my_collate, num_workers = 5)
    test_dataloader = DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 8)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader, ckpt_path = "best")

    i += 1