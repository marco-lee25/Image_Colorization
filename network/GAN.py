import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from torchsummary import summary
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
from .Discriminator import Critic
from .Generator import Generator

import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def display_progress(cond, real, fake, current_epoch = 0, figsize=(20,15)):
    """
    Save cond, real (original) and generated (fake)
    images in one panel 
    """
    cond = cond.detach().cpu().permute(1, 2, 0)   
    real = real.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)
    
    images = [cond, real, fake]
    titles = ['input','real','generated']
    print(f'Epoch: {current_epoch}')
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for idx,img in enumerate(images):
        if idx == 0:
            ab = torch.zeros((224,224,2))
            img = torch.cat([images[0]* 100, ab], dim=2).numpy()
            imgan = lab2rgb(img)
        else:
            imgan = lab_to_rgb(images[0],img)
        ax[idx].imshow(imgan)
        ax[idx].axis("off")
    for idx, title in enumerate(titles):    
        ax[idx].set_title('{}'.format(title))
    
    fig.savefig("./result/"+str(current_epoch)+".jpg")

def lab_to_rgb(L, ab):
    """
    Takes an image or a batch of images and converts from LAB space to RGB
    """
    L = L  * 100
    ab = (ab - 0.5) * 128 * 2
    Lab = torch.cat([L, ab], dim=2).numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


# class CWGAN(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=100, display_step=10, lambda_gp=10, lambda_r1=10, device='cpu'):
#         super().__init__()

#         # self.save_hyperparameters()
        
#         self.display_step = display_step
        
#         self.generator = Generator(in_channels, out_channels).to(device)
#         self.critic = Critic(in_channels + out_channels).to(device)

#         self.optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
#         self.optimizer_C = optim.Adam(self.critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))

#         self.lambda_recon = lambda_recon

#         self.lambda_gp = lambda_gp

#         self.lambda_r1 = lambda_r1

#         self.recon_criterion = nn.L1Loss().to(device)
#         self.generator_losses, self.critic_losses  =[],[]

#     def configure_optimizers(self):
#         return [self.optimizer_C, self.optimizer_G]

#     def generator_step(self, real_images, conditioned_images):
#         # WGAN has only a reconstruction loss
#         self.optimizer_G.zero_grad()
#         fake_images = self.generator(conditioned_images)
#         recon_loss = self.recon_criterion(fake_images, real_images)
#         recon_loss.backward()
#         self.optimizer_G.step()
#         # Keep track of the average generator loss
#         self.generator_losses += [recon_loss.item()]

#     def critic_step(self, real_images, conditioned_images):
#         self.optimizer_C.zero_grad()
#         fake_images = self.generator(conditioned_images)
#         fake_logits = self.critic(fake_images, conditioned_images)
#         real_logits = self.critic(real_images, conditioned_images)
#         # Compute the loss for the critic
#         loss_C = real_logits.mean() - fake_logits.mean()

#         # Compute the gradient penalty
#         alpha = torch.rand(real_images.size(0), 1, 1, 1, requires_grad=True)
#         alpha = alpha.to(device)
 
#         interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)
        
#         interpolated_logits = self.critic(interpolated, conditioned_images)
        
#         grad_outputs = torch.ones_like(interpolated_logits, dtype=torch.float32, requires_grad=True)
#         gradients = torch.autograd.grad(outputs=interpolated_logits, inputs=interpolated, grad_outputs=grad_outputs,create_graph=True, retain_graph=True)[0]

#         gradients = gradients.view(len(gradients), -1)
#         gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#         loss_C += self.lambda_gp * gradients_penalty
        
#         # Compute the R1 regularization loss
#         r1_reg = gradients.pow(2).sum(1).mean()
#         loss_C += self.lambda_r1 * r1_reg

#         # Backpropagation
#         loss_C.backward()
#         self.optimizer_C.step()
#         self.critic_losses += [loss_C.item()]

#     def get_loss(self):
#         gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
#         crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step
#         return gen_mean, crit_mean

#     def forward(self, batch, batch_idx, optimizer_idx, epoch):
#         real, condition = batch
#         if optimizer_idx == 0:
#             self.critic_step(real, condition)
#         elif optimizer_idx == 1:
#             self.generator_step(real, condition)

#         gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
#         crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step
#         if batch_idx%20 == 0:
#             print(f"Epoch {epoch} : Batch id : {batch_idx}, Generator loss: {gen_mean}, Critic loss: {crit_mean}")

#         if batch_idx%450 == 0:
#             fake = self.generator(condition).detach()
#             print("save figure")
#             display_progress(condition[0], real[0], fake[0], epoch)
    
#         if epoch & 30 == 0:
#             torch.save(self.generator.state_dict(), "ResUnet_"+ str(epoch) +".pt")
#             torch.save(self.critic.state_dict(), "PatchGAN_"+ str(epoch) +".pt")

#         # if epoch%self.display_step==0 and batch_idx==0 and optimizer_idx==1:
#         #     fake = self.generator(condition).detach()
#         #     torch.save(self.generator.state_dict(), "ResUnet_"+ str(epoch) +".pt")
#         #     torch.save(self.critic.state_dict(), "PatchGAN_"+ str(epoch) +".pt")
#         #     print(f"Epoch {epoch} : Generator loss: {gen_mean}, Critic loss: {crit_mean}")

class CWGAN(pl.LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=100, display_step=10, lambda_gp=10, lambda_r1=10,):

        super().__init__()
        self.save_hyperparameters()
        
        self.display_step = display_step
        
        self.generator = Generator(in_channels, out_channels)
        self.critic = Critic(in_channels + out_channels)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.optimizer_C = optim.Adam(self.critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.lambda_recon = lambda_recon
        self.lambda_gp = lambda_gp
        self.lambda_r1 = lambda_r1
        self.recon_criterion = nn.L1Loss()
        self.generator_losses, self.critic_losses  =[],[]
    
    def configure_optimizers(self):
        return [self.optimizer_C, self.optimizer_G]
        
    def generator_step(self, real_images, conditioned_images):
        # WGAN has only a reconstruction loss
        self.optimizer_G.zero_grad()
        fake_images = self.generator(conditioned_images)
        recon_loss = self.recon_criterion(fake_images, real_images)
        recon_loss.backward()
        self.optimizer_G.step()
        
        # Keep track of the average generator loss
        self.generator_losses += [recon_loss.item()]
        
        
    def critic_step(self, real_images, conditioned_images):
        self.optimizer_C.zero_grad()
        fake_images = self.generator(conditioned_images)
        fake_logits = self.critic(fake_images, conditioned_images)
        real_logits = self.critic(real_images, conditioned_images)
        
        # Compute the loss for the critic
        loss_C = real_logits.mean() - fake_logits.mean()

        # Compute the gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1, requires_grad=True)
        alpha = alpha.to(device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)
        
        interpolated_logits = self.critic(interpolated, conditioned_images)
        
        grad_outputs = torch.ones_like(interpolated_logits, dtype=torch.float32, requires_grad=True)
        gradients = torch.autograd.grad(outputs=interpolated_logits, inputs=interpolated, grad_outputs=grad_outputs,create_graph=True, retain_graph=True)[0]

        
        gradients = gradients.view(len(gradients), -1)
        gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss_C += self.lambda_gp * gradients_penalty
        
        # Compute the R1 regularization loss
        r1_reg = gradients.pow(2).sum(1).mean()
        loss_C += self.lambda_r1 * r1_reg

        # Backpropagation
        loss_C.backward()
        self.optimizer_C.step()
        self.critic_losses += [loss_C.item()]
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        real, condition = batch
        if optimizer_idx == 0:
            self.critic_step(real, condition)
        elif optimizer_idx == 1:
            self.generator_step(real, condition)
        gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
        crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step
        if self.current_epoch%self.display_step==0 and batch_idx==0 and optimizer_idx==1:
            fake = self.generator(condition).detach()
            torch.save(self.generator.state_dict(), "ResUnet_"+ str(self.current_epoch) +".pt")
            torch.save(self.critic.state_dict(), "PatchGAN_"+ str(self.current_epoch) +".pt")
            print(f"Epoch {self.current_epoch} : Generator loss: {gen_mean}, Critic loss: {crit_mean}")
            display_progress(condition[0], real[0], fake[0], self.current_epoch)