import os 
import numpy as np

def get_data(ab_path = "./ab/ab/ab1.npy", l_path = "./l/gray_scale.npy"):
    ab_df = np.load(ab_path)[0:5000]
    L_df = np.load(l_path)[0:5000]
    dataset = (L_df,ab_df )
    gc.collect()

    return dataset