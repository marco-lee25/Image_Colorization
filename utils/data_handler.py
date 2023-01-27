import os 
import numpy as np
import torch

def get_data(ab_path = "./ab/ab/ab1.npy", l_path = "./l/gray_scale.npy"):
    ab_df = np.load(ab_path)[0:5000]
    L_df = np.load(l_path)[0:5000]
    dataset = (L_df,ab_df )
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