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
import argparse

from model import Generator, Discriminator
from utils import save_image_grid

class GAN:
    def __init__(self, batch_size=64, sample_size=100, g_alpha=0.01, g_lr=1.0e-3, d_alpha=0.01, d_lr=1.0e-4):
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.g_alpha = g_alpha
        self.g_lr = g_lr
        self.d_alpha = d_alpha
        self.d_lr = d_lr

        # loading mnist dataset from torch library
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)

        # Real and fake labels
        self.real_targets = torch.ones(self.batch_size, 1)
        self.fake_targets = torch.zeros(self.batch_size, 1)

        # Generator and discriminator networks
        self.generator = Generator(self.sample_size, self.g_alpha)
        self.discriminator = Discriminator(self.d_alpha)

        # Optimizers
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr)

    def train_discriminator(self, images, real_targets, fake_targets):
        # Loss with MNIST image inputs and real_targets as labels
        self.discriminator.train()
        d_loss = self.discriminator(images, real_targets)
        
        # Generate images in eval mode
        self.generator.eval()
        with torch.no_grad():
            generated_images = self.generator(self.batch_size)

        # Loss with generated image inputs and fake_targets as labels
        d_loss += self.discriminator(generated_images, fake_targets)

        # Optimizer updates the discriminator parameters
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item()

    def train_generator(self, real_targets):
        # Generate images in train mode
        self.generator.train()
        generated_images = self.generator(self.batch_size)

        # Loss with generated image inputs and real_targets as labels
        g_loss = self.discriminator(generated_images, real_targets)

        # Optimizer updates the generator parameters
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item()

    def train(self):
        for epoch in range(100):
            d_losses = []
            g_losses = []

            for images, labels in tqdm(self.dataloader):
                d_loss = self.train_discriminator(images, self.real_targets, self.fake_targets)
                g_loss = self.train_generator(self.real_targets)

                # Keep losses for logging
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                
            # Print average losses
            print(epoch, np.mean(d_losses), np.mean(g_losses))

            # Save images
            save_image_grid(epoch, self.generator(self.batch_size), ncol=8)
if __name__=="__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--sample_size', type=int, default=100, help='Sample size for the generator')
    parser.add_argument('--g_alpha', type=float, default=0.01, help='Alpha for the generator')
    parser.add_argument('--g_lr', type=float, default=1.0e-3, help='Learning rate for the generator')
    parser.add_argument('--d_alpha', type=float, default=0.01, help='Alpha for the discriminator')
    parser.add_argument('--d_lr', type=float, default=1.0e-4, help='Learning rate for the discriminator')

    # Parse the arguments
    args = parser.parse_args()
    
    gan = GAN(**vars(args))
    gan.train()