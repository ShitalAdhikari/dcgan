import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch

# function to save the image
def save_image_grid(epoch: int, images: torch.Tensor, ncol: int, file_dir = "data"):
    os.makedirs(file_dir, exist_ok=True)
    image_grid = make_grid(images, ncol)     # Images in a grid
    image_grid = image_grid.permute(1, 2, 0) # Move channel last
    image_grid = image_grid.cpu().numpy()    # To Numpy

    plt.imshow(image_grid)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{file_dir}/generated_{epoch:03d}.jpg')
    plt.close()