import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

# Objective:
# 1. Learn how to apply an activation function.
# 2. Learn about max pooling

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
Gx = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0, -1.0]])
conv.state_dict()['weight'][0][0] = Gx
conv.state_dict()['bias'][0] = 0.0
print(conv.state_dict())

image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1

Z = conv(image)
# Apply the activation function to the activation map. This will apply
# the activation function to each element in the activation map.
A = torch.relu(Z)

# Max Pooling
image1 = torch.zeros(1, 1, 4, 4)
image1[0, 0, 0, :] = torch.tensor([1.0, 2.0, 3.0, -4.0])
image1[0, 0, 1, :] = torch.tensor([0.0, 2.0, -3.0, 0.0])
image1[0, 0, 2, :] = torch.tensor([0.0, 2.0, 3.0, 1.0])

# Create a maxpooling object in 2d
max1 = torch.nn.MaxPool2d(2, stride=1)
max1(image1)
# If the stride is set to None (its defaults setting), the process will simply take
# the maximum in a prescribed area and shift over accordingly
max1 = torch.nn.MaxPool2d(2)
max1(image1)
