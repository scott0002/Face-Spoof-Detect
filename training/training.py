from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
plt.ion()   # interactive mode

def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
    model.eval()

def load_image():
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'development': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = "..\LCC_FASD"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['training', 'development']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['training', 'development']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'development']}
    class_names = image_datasets['training'].classes
    #print(class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    return class_names, dataloaders

def imshow(class_names, dataloaders, inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated
    # Get a batch of training data
    

def training():
    class_names, dataloaders=load_image()
    


if __name__ == '__main__':
    training()