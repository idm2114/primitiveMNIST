import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from neuralnet import NeuralNetwork
import os
import cv2
import re
import numpy as np
from skimage import io, transform

class MNISTdataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # self.img_list = [name for name in os.listdir(self.root_dir) if os.path.isfile(name)]
        self.img_list = [name for name in os.listdir(self.root_dir)] 
        self.img_list = sorted(self.img_list, key=lambda name : re.search(r'\d+', name).group()) 
        print(self.img_list)

    def __len__(self):
        return(len(self.img_list))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.img_list[idx])
        image = io.imread(img_name)
        # resizing image
        desired_size = 28 
        old_size = image.shape[:2] # old_size is in (height, width) format
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        sample = cv2.resize(image, (new_size[1], new_size[0]))
        cv2.imshow("imageresize", sample) 
        cv2.waitKey(0)

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [255, 255, 255]
        new_im = cv2.copyMakeBorder(sample, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        cv2.imshow("newim", new_im)
        #sample = np.asarray(image)
        #sample = cv2.resize(sample, (new_size[1], new_size[0]))
        #cv2.imshow("sample", sample)
        cv2.waitKey(0)

        new_im = (255 - new_im)
        cv2.imshow("newimInv", new_im) 
        cv2.waitKey(0)
        
        if self.transform:
            new_im = self.transform(new_im)


        return new_im 
