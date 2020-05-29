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
plt.ion() 

def load_image():
    data_transforms = {
        'evaluation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = "..\LCC_FASD"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['evaluation']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['evaluation']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['evaluation']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    return dataset_sizes, device, class_names, dataloaders

def run_test():
    model = models.resnet18(pretrained=True)
    model.load_state_dict(torch.load(model))
    running_loss = 0.0
    running_corrects = 0
    model.eval()

if __name__ == '__main__':
    run_test()