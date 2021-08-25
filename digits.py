import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets 
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda, Compose, Normalize
import matplotlib.pyplot as plt
from neuralnet import NeuralNetwork

class digitClassifier(): 
    def __init__(self):

        self.train_batch_size = 64 
        self.test_batch_size = 1000 

        # initialize datasets (only done once)
        self.train_data = datasets.MNIST(
                root="data", 
                train=True,
                download=True,
                transform=Compose([
                    ToTensor(),
                    Normalize((.1307,), (.3081,)) ])
            )
        self.test_data = datasets.MNIST(
                root="data", 
                train=False,
                download=True,
                transform=Compose([
                    ToTensor(),
                    Normalize((.1307,), (.3081,)) ])
            )


        self.train_loaded = DataLoader(self.train_data, batch_size = self.train_batch_size, shuffle=True)
        self.test_loaded = DataLoader(self.test_data, batch_size = self.test_batch_size, shuffle=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = NeuralNetwork().to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        # using stochastic gradient descent as optimizer
        # gradient(F) = <F_x, F_y, F_z> in 3d 
        # F = 3x^2y^2z^2
        # grad(F) (often denoted as upside down triangle)
        # grad(F) = <6xy^2z^2, 6x^2yz^2, 6x^2y^2z>

        # F = x^2 + y^2 + z^2
        # grad(F) (often denoted as upside down triangle)
        # grad(F) = <dF/dx, dF/dy, dF/dz> = <2x, 2y, 2z>
        # serves as an approximation for the way a function is changing

        # gradient(F) = <F_alpha1, F_alpha2, F_alpha3, ...> in higher order case 

        # play with learning rate number later to observe variance 
        # learning rate = .01
        # prev: learning rate = .001 (1e-3)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-2, momentum=.5)

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []


    def train(self, epoch):
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.train_loaded):
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_loaded.dataset),
                100. * batch_idx / len(self.train_loaded), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(self.train_loaded.dataset)))
                torch.save(self.net.state_dict(), 'results/model.pth')
                torch.save(self.optimizer.state_dict(), 'results/optimizer.pth')

    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loaded:
                output = self.net(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loaded.dataset)
        self.test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(self.test_loaded.dataset),
        100. * correct / len(self.test_loaded.dataset)))

    def run(self, epochs):
        self.test()
        for i in range(epochs):
            print(f"starting epoch {i+1}")
            self.train(i)
            self.test()

    def save(self, path):
        torch.save(self.net.state_dict(), f"{path}.pth")
        print(f"saved model to {path}.pth")

if __name__=="__main__":
    clf = digitClassifier()
    clf.run(10)
    clf.save("classifier")
