import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, Normalize
import matplotlib.pyplot as plt
from neuralnet import NeuralNetwork
from MNISTdataset import MNISTdataset 

class networkTest():
    def __init__(self):
        # storing a random phone number as an array in python
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_data = None

    def set_test_files(self, imgdir):
        self.test_data = MNISTdataset(imgdir, transform=Compose([
                        ToTensor(),
                        Normalize((.1307,), (.3081,)) ])
            )

    def load_classifier(self, path):
        self.net = NeuralNetwork()
        self.net.load_state_dict(torch.load(f'{path}.pth'))

    def load_image(self, img):
        img = self.imagedir
        batch_size = 1 
        test_loaded = DataLoader(test_data, batch_size = 1)

    def test(self):
        # should call on the classifier for each image in the image directory
        for itm in self.test_data:
            pred_out, pred_index = torch.max(self.net(itm[None, ...]),1)
            print(pred_index[-1].item())

if __name__=="__main__":
    n = networkTest()
    n.load_classifier("classifier")
    n.set_test_files("test_images")
    n.test()

