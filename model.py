import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm


# Generator network with transposed convolutions
class Generator(nn.Module):
    def __init__(self, sample_size: int, alpha: float):
        super().__init__()
        # sample_size => 784 

        self.alpha = alpha
        
        self.l1 = nn.Linear(sample_size, 784)
        self.bn1 = nn.BatchNorm1d(784)

        #
        self.conv1 = nn.ConvTranspose2d(16, 32, 
                               kernel_size=5, stride=2, padding=2,
                               output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.ConvTranspose2d(32, 1,
                               kernel_size=5, stride=2, padding=2,
                               output_padding=1, bias=False)

        # Random value vector size
        self.sample_size = sample_size
        

    def forward(self, batch_size: int):
        # Random value generation
        z = torch.randn(batch_size, self.sample_size)
        # print(z.shape)

        # the linear layer extracts the 784 features
        x = nn.LeakyReLU(self.alpha)(self.bn1(self.l1(z)))
        # print(x.shape)

        # the 784 vector size is reshaped to 16*7*7, where 7*7 acts as the image size and 16 as the number of features
        x = torch.reshape(x, (-1, 16,7,7)) 

        # 16*7*7 is converted to the 32*14*14. 14*14 scale up image comes from the increased kernel size of 5 during TransposeConvolution
        x = nn.LeakyReLU(self.alpha)(self.bn2(self.conv1(x)))   

        # 
        # print(x.shape)

        # again the transpose convolution is used to convert change image size to 28*28 with 1 features.
        # application of sigmoid fixes the layer output to the range of the 0 to 1
        x = nn.Sigmoid()(self.conv2(x))
        
        return x

# Discriminator network with convolutions
# the goal of the descriminator is used to classify whether the image is real or fake and is tested
class Discriminator(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32,
                    kernel_size=5, stride=2, padding=2, bias=False)
        
        self.conv2 = nn.Conv2d(32, 16,
                    kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.l1 = nn.Linear(784, 784)
        self.bn2 = nn.BatchNorm1d(784)
        
        self.l2 = nn.Linear(784, 1)
        self.alpha = alpha

    def forward(self, images: torch.Tensor, targets: torch.Tensor):

        # the input image is convolved and the image size is also reduced during the convolution without the use of maxpooling
        # the paper mentions the removal of the max pooling layer with the convolution layer
        # this further makes the model learn the parameters even during the pooling operation

        # given the input image is 28 * 28 * 1 -> output size of 32 * 14 * 14 is extracted
        x = nn.LeakyReLU(self.alpha)(self.conv1(images))  
        

        # further the size is reduced to 16 * 7 * 7
        x = nn.LeakyReLU(self.alpha)(self.bn1(self.conv2(x)))       
        

        # the input is flattened to 784 size then fed to linear layer
        x = nn.Flatten()(x)
        
        x = self.bn2(self.l1(x))
        
        x = nn.LeakyReLU(self.alpha)(x)
        
        # finally the single dimension output is extracted.
        prediction = self.l2(x) 

        # bcelogits is returned as the loss
        loss = F.binary_cross_entropy_with_logits(prediction, targets)
        return loss