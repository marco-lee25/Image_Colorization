import os
from pathlib import Path
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import PIL
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.models.resnet import resnet18
from torchvision.models.vgg import vgg19
from torch.utils.data import Dataset, DataLoader
# from ..config import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    '''
    
    The Discriminator Model Class
    
    '''
    
    def __init__(self, Config, input_channels,num_filters=64, n_down=3):
        super().__init__()
        self.Config = Config
        model = [self.get_layers(input_channels,num_filters,norm = False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), stride=1 if i == (n_down-1) else 2) for i in range(n_down)]
        model += [self.get_layers(num_filters * 2 ** n_down, 1, stride=1, norm=False, activation=False)] 
        self.model = nn.Sequential(*model)
    
    def get_layers(self, ni, nf): 
        kernel_size=self.Config.kernel_size
        stride=self.Config.stride
        padding=self.Config.padding
        LeakyReLU_slope = self.Config.LeakyReLU_slope

        norm=True
        activation=True

        layers = [nn.Conv2d(ni, nf, kernel_size, stride, padding, bias=not norm)]          
        if norm: 
            layers += [nn.BatchNorm2d(nf)]
        if activation:
            layers += [nn.LeakyReLU(LeakyReLU_slope, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
