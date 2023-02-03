import os 
import numpy as np
import torch
import gc
import PIL
from PIL import Image
from config import Config
from torch.utils.data import Dataset
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data(ab_path = "./data/ab/ab/ab1.npy", l_path = "./data/l/gray_scale.npy"):
    ab_df = np.load(ab_path)[0:5000]
    L_df = np.load(l_path)[0:5000]
    dataset = (L_df,ab_df)
    gc.collect()
    return dataset

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

class ImageColorizationDataset(Dataset):
    ''' Black and White (L) Images and corresponding A&B Colors'''
    def __init__(self, dataset, transform=None):
        '''
        :param dataset: Dataset name.
        :param data_dir: Directory with all the images.
        :param transform: Optional transform to be applied on sample
        '''
        self.dataset = dataset
        self.data = transforms.ToTensor()(dataset[0].reshape((224, 224, len(dataset[0])))).to(device)
        self.target = dataset[1]

        self.data = []
        self.target = []
        for id in range(len(dataset[0])):
            L = np.array(self.dataset[0][id]).reshape((224,224,1))
            L = transforms.ToTensor()(L)
            self.data.append(L)

        for id in range(len(dataset[1])):
            ab = np.array(self.dataset[1][id])
            ab = transforms.ToTensor()(ab)
            self.target.append(ab)
        
        # for x in dataset[1]:
        #      self.target.append(x)
        # tmp = []
        # for i in dataset[1]:
        #     tmp.append(transforms.ToTensor()(i))
        # tmp = torch.from_numpy(np.array(tmp))
        # print(tmp[0].shape)
        # exit()
        # self.target = np.array(self.target)
        # self.target = torch.from_numpy(dataset[1]).to(device)
        

        # self.target = torch.from_numpy(dataset[1]).to(device)
        
        # self.data =  torch.from_numpy(dataset[0])
        # self.target = torch.from_numpy(dataset[1])
        self.transform = transform

    def __len__(self):
        return len(self.dataset[0])
    
    def __getitem__(self, idx):
        # L = np.array(self.dataset[0][idx]).reshape((224,224,1))
        # L = transforms.ToTensor()(L)
        # ab = np.array(self.dataset[1][idx])
        # ab = transforms.ToTensor()(ab)

        L = self.data[idx]
        ab = self.target[idx]
        return ab, L

class ImageDataset(Dataset):
    ''' 
    
    Class that deals with the Data Loading and preprocessing steps such as image resizing, data augmentation (horizontal flip) 
    and conversion of RGB image to LAB color space with standardization.
    
    '''
    
    def __init__(self,paths,train = True):
        if train == True:
            self.transforms = transforms.Compose([transforms.Resize((Config.image_size_1,Config.image_size_2)),
                                                 transforms.RandomHorizontalFlip()]) # Basic Data Augmentation
        elif train == False:
            self.transforms = transforms.Compose([transforms.Resize((Config.image_size_1,Config.image_size_2))])
            
        self.train = train
        self.paths = paths
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        lab = rgb2lab(img).astype("float32")
        lab = transforms.ToTensor()(lab)
        L = lab[[0],...]/50 - 1 # Standardizing L space
        ab = lab[[1,2],...]/128 # Standardizing ab space
        
        return {'L': L,