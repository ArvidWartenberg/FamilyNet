import time
import tqdm
import matplotlib.pyplot as plt
import torch
from constants import *
from sklearn.metrics import average_precision_score as AP

class Trainer():

    @staticmethod
    def train_model(model, dataloaders, dataset_sizes, objective, optimizer, device, num_epochs=25):

        # Prepare
        since = time.time()
        best_model_wts = model.state_dict()
        best_loss = 999999
        running_losses = {'train': [], 'valid': []}
        running_aps = {'train': {'all': [], 'Arvid': [], 'Sofia': [], 'Fredrik': [], 'Constanze': []},
                        'valid': {'all': [], 'Arvid': [], 'Sofia': [], 'Fredrik': [], 'Constanze': []}}

        # Loop
        for epoch in range(num_epochs):

            # Progress
            print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            outputs_conc = np.array([])
            labels_conc = np.array([])
            for split in ['train', 'valid']:
                torch.set_grad_enabled(split == 'train')
                model.train(split=='train')
                total_loss = 0.0

                # Load and train/eval on split
                for data in tqdm.tqdm(dataloaders[split],colour='ffffff'):

                    # Put batch on device
                    inputs, labels = data['image'].to(device), data['label'].to(device)

                    # Forward and calc loss
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = objective(outputs, labels)
                    # Backprop if train
                    if split == 'train':
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()
                    outputs_conc = np.concatenate((outputs.detach().cpu().numpy(),outputs.detach().cpu().numpy()),axis=0)
                    labels_conc = np.concatenate((labels.detach().cpu().numpy(),labels.detach().cpu().numpy()),axis=0)


                # Record
                epoch_loss = total_loss / dataset_sizes[split]
                running_losses[split].append(epoch_loss)
                running_aps[split]['all'].append(AP(labels_conc,outputs_conc))
                for name in list(name_to_class.keys()):
                    running_aps[split][name].append(AP(labels_conc[:,name_to_class[name]],outputs_conc[:,name_to_class[name]]))
                print('{} Loss: {:.4f}'.format(split, epoch_loss))

                # Deep copy if best loss
                if split == 'valid' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()



            # Plot results
            plt.subplot(2,1,1)
            plt.title('Losses/Metrics', fontsize=15)
            plt.plot(running_losses['train'], label='Training loss',color='blue',linewidth=3)
            plt.plot(running_losses['valid'], label='Validation loss',color='red',linewidth=3)
            plt.legend()
            plt.ylabel('Loss', fontsize=13)
            plt.subplot(2,1,2)
            for name in ['all']:#,*list(name_to_class.keys())]:
                plt.plot(running_aps['train'][name], label='Train AP ' + name,linewidth=3,linestyle=':')
                plt.plot(running_aps['valid'][name], label='Val AP ' + name,linewidth=3)
            plt.ylabel('AP', fontsize=13)
            plt.xlabel('Epoch', fontsize=13)
            plt.legend()
            plt.savefig('train_fig')
            plt.close()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best Validation Loss: {:4f}'.format(best_loss))


        # load best model weights
        model.load_state_dict(best_model_wts)
        return model