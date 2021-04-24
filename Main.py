import torch
import torchvision
import pandas as pd
from torch.utils.data import DataLoader
from Dataset import ImageDataset
from Model import FamilyNet
from Trainer import Trainer
from constants import *

'''

Inbalance in classes, fix by assigning weights in BCE loss.

'''


if __name__ == '__main__':

    # Use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Load annotations
    annotations_df = pd.read_csv('annotations.csv')

    # Debug
    #annotations_df = annotations_df.iloc[0:100]

    # Dataloader
    datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    for split in ['train', 'valid', 'test']:
        datasets[split] = ImageDataset(df=annotations_df,
                                       split=split,
                                       im_size=299)

        dataloaders[split] = DataLoader(datasets[split],
                                        batch_size=32,
                                        shuffle=(split != 'test'),
                                        num_workers=4,
                                        pin_memory=True)

        dataset_sizes[split] = len(datasets[split])

    # Init model
    model = FamilyNet(n_classes=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    objective = torch.nn.BCELoss()

    trained_model = Trainer.train_model(model=model,
                                        dataloaders=dataloaders,
                                        dataset_sizes=dataset_sizes,
                                        objective=objective,
                                        optimizer=optimizer,
                                        device=device,
                                        num_epochs=25)

    torch.save(trained_model.state_dict(),'FamilyNet.pth')