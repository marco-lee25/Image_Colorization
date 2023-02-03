from . import config
from utils import data_handler
from torch.utils.data import DataLoader
from network.Generator import Generator
from network.Discriminator import Critic
from skimage.color import rgb2lab, lab2rgb
from torchsummary import summary
from torch.utils.data.dataloader import default_collate
# from network.GAN import CWGAN
from pathlib import Path
from PIL import Image
from fastai.data.external import untar_data, URLs
from network import trainer
import PIL
# import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
import torch
import logging
import time
import glob

LOG = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    path = untar_data(URLs.COCO_SAMPLE)
    path = str(path) + "/train_sample"
    paths = glob.glob(path + "/*.jpg")

    # Train test split
    np.random.seed(42)
    paths_subset = np.random.choice(paths, config.Config.external_data_size, replace=False) # choosing 10000 images randomly
    random_idxs = np.random.permutation(config.Config.external_data_size) 

    train_idxs = random_idxs[:config.Config.train_size] # choosing the first 8000 as training set
    val_idxs = random_idxs[config.Config.train_size:] # choosing last 2000 as validation set
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    
    _, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, img_path in zip(axes.flatten(), train_paths):
        ax.imshow(Image.open(img_path))
        ax.axis("off")

    #train dataset
    train_data = data_handler.ImageDataset(paths = train_paths, train=True)
    # validation dataset
    valid_data = data_handler.ImageDataset(paths = val_paths, train=False)
    # train data loader
    train_loader = DataLoader(train_data, batch_size=config.batch_size,shuffle=True,pin_memory = True)
    # validation data loader
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size,shuffle=False,pin_memory = True

    config = config.config
    trainer = trainer.model_trainer(model, train_loader, 1, 2, config)

    generator = build_backbone_unet(input_channels=1, output_channels=2, size=Config.image_size_1)
    opt = optim.Adam(generator.parameters(), lr=Config.pretrain_lr)
    loss = nn.L1Loss()        
    pretrain_generator(generator, train_loader, opt, loss, Config.epochs)
    torch.save(generator.state_dict(), "res18-unet.pt")
    generator.load_state_dict(torch.load("res18-unet.pt", map_location=device))
    model = MainModel(generator=generator)
    train_model(model, train_loader, Config.epochs)
    torch.save(model.state_dict(), "main-model.pt")