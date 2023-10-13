import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import os

from model import Generator, Discriminator
from utils import save_image_grid

# General config
batch_size = 64

# Generator config
sample_size = 100    
g_alpha     = 0.01   
g_lr        = 1.0e-3 

# Discriminator config
d_alpha = 0.01       
d_lr    = 1.0e-4   

# loading mnist datgaset from torch library
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

# Real and fake labels
real_targets = torch.ones(batch_size, 1)
fake_targets = torch.zeros(batch_size, 1)

# Generator and discriminator networks
generator = Generator(sample_size, g_alpha)
discriminator = Discriminator(d_alpha)

# Optimizers
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)


def train_descriminator():

    # Loss with MNIST image inputs and real_targets as labels
    discriminator.train()
    d_loss = discriminator(images, real_targets)
    
    # Generate images in eval mode
    generator.eval()
    with torch.no_grad():
        generated_images = generator(batch_size)

    # Loss with generated image inputs and fake_targets as labels
    d_loss += discriminator(generated_images, fake_targets)

    # Optimizer updates the discriminator parameters
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    return d_loss.item()

def train_generator():

    # Generate images in train mode
    generator.train()
    generated_images = generator(batch_size)

    # batchnorm is unstable in eval due to generated images
    # change drastically every epoch. We'll not use the eval here.
    # discriminator.eval() 

    # Loss with generated image inputs and real_targets as labels
    g_loss = discriminator(generated_images, real_targets)

    # Optimizer updates the generator parameters
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item()

for epoch in range(100):

    d_losses = []
    g_losses = []

    for images, labels in tqdm(dataloader):

        d_loss = train_descriminator()
        g_loss = train_generator()

        # Keep losses for logging
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
    # Print average losses
    print(epoch, np.mean(d_losses), np.mean(g_losses))

    # Save images
    save_image_grid(epoch, generator(batch_size), ncol=8)