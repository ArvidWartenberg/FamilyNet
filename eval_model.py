import torch
import torchvision
import pandas as pd
from torch.utils.data import DataLoader
from Dataset import ImageDataset
from Model import FamilyNet
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from PIL import Image
from Trainer import Trainer
from constants import *
import tqdm
import os

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
    for split in ['valid']:
        datasets[split] = ImageDataset(df=annotations_df,
                                       split=split,
                                       im_size=299)

        dataloaders[split] = DataLoader(datasets[split],
                                        batch_size=9,
                                        shuffle=(split != 'test'),
                                        num_workers=4,
                                        pin_memory=True)

        dataset_sizes[split] = len(datasets[split])

    model = FamilyNet(n_classes=4).to(device)
    model.load_state_dict(torch.load('FamilyNet.pth'))

    torch.set_grad_enabled(False)
    model.train(False)
    total_loss = 0.0

    # Load and train/eval on split
    for data in tqdm.tqdm(dataloaders['valid']):

        # Put batch on device
        inputs, labels = data['image'].to(device), data['label'].to(device)

        # Forward and calc loss
        outputs = model(inputs)
        np_out = outputs[:][:].detach().cpu().numpy()
        plt.figure()
        for b_ix in range(0,9):
            plt.subplot(3,3,b_ix+1)
            plt.imshow(to_pil_image(data['image'][b_ix]))
            plt.xticks([])
            plt.yticks([])
            plt.title('A:%.2f, S:%.2f, F:%.2f, C:%.2f, '%(np_out[b_ix][0],np_out[b_ix][1],np_out[b_ix][2],np_out[b_ix][3]),fontsize=7)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.draw()
        n_files = len(os.listdir('plots'))
        plt.savefig('plots/model_1_fig_'+str(n_files+1),dpi=300)
        plt.close()
