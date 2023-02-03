from config import Config
from utils import data_handler
from torch.utils.data import DataLoader
from network.Generator import Generator
from network.Discriminator import Critic
from skimage.color import rgb2lab, lab2rgb
from torchsummary import summary
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
# from network.GAN import CWGAN
from pathlib import Path
from PIL import Image
from fastai.data.external import untar_data, URLs
from network import trainer
from utils.data_handler import lab_to_rgb
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
    paths_subset = np.random.choice(paths, Config.external_data_size, replace=False) # choosing 10000 images randomly
    random_idxs = np.random.permutation(Config.external_data_size) 

    train_idxs = random_idxs[:Config.train_size] # choosing the first 8000 as training set
    val_idxs = random_idxs[Config.train_size:] # choosing last 2000 as validation set
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    
    _, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, img_path in zip(axes.flatten(), train_paths):
        ax.imshow(Image.open(img_path))
        ax.axis("off")

    #train dataset
    train_data = data_handler.ImageDataset(paths = train_paths, config, train=True)
    # validation dataset
    valid_data = data_handler.ImageDataset(paths = val_paths, config, train=False)
    # train data loader
    train_loader = DataLoader(train_data, batch_size=Config.batch_size,shuffle=True,pin_memory = True)
    # validation data loader
    valid_loader = DataLoader(valid_data, batch_size=Config.batch_size,shuffle=False,pin_memory = True)

    trainer = trainer.model_trainer(train_loader, 1, 2, Config)
    trainer.train_model()

    model = trainer.get_model()
    path = "Path to Image"
    img = PIL.Image.open(path)
    img = img.resize((256, 256))
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        preds = model.generator(img.unsqueeze(0).to(device))
        gen_output = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    plt.imshow(img)
    plt.imshow(gen_output)