from utils import data_handler, config
from torch.utils.data import DataLoader
from network.Generator import Generator
from network.Discriminator import Critic
from skimage.color import rgb2lab, lab2rgb
from torchsummary import summary
from torch.utils.data.dataloader import default_collate
from network.GAN import CWGAN

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
import torch
import logging
import time
LOG = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    data = data_handler.get_data()
    L_df, ab_df = data[0], data[1]

    batch_size = 10

    # Prepare the Datasets
    train_dataset = data_handler.ImageColorizationDataset((L_df, ab_df))
    test_dataset = data_handler.ImageColorizationDataset((L_df, ab_df))
    
    # Build DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True, pin_memory = False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle = False, pin_memory = False)

    cwgan = CWGAN(in_channels = 1, out_channels = 2 ,learning_rate=2e-4, lambda_recon=100, display_step=10)#, device=device)

    # cwgan.train()

    # optimizer_idx = 0

    # for epoch in range(150):
    #     epoch_loss = 0.0
    #     for batch_idx, (im, target) in enumerate(train_loader):
    #         data = (im.to(device, non_blocking=True), target.to(device, non_blocking=True))
    #         if optimizer_idx == 0:
    #             optimizer_idx = 1
    #         else:
    #             optimizer_idx = 0

    #         output = cwgan(data, batch_idx,  optimizer_idx, epoch)
    #         g_losses, c_losses = cwgan.get_loss()

    #         if batch_idx % 2 == 0:
    #             batch_info = {
    #                 'type': 'train',
    #                 'epoch': epoch, 'batch': batch_idx, 'n_batches': len(data),
    #                 'g_loss': round(g_losses, 3) ,
    #                 'c_loss': round(c_losses, 3) 
    #             }
    #             LOG.info(batch_info)
    #     torch.cuda.synchronize()

    trainer = pl.Trainer(max_epochs=150, gpus=-1)
    trainer.fit(cwgan, train_loader)