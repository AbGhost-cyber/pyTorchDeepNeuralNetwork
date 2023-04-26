import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

# Objective:
# 1. Learn about Convolution.
# 2. Learn Determining the Size of Output.
# 3. Learn Stride, Zero Padding

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
# since the parameters are randomly initialized and learning during training,
# we can give them some values
conv.state_dict()['weight'][0][0] = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0.0, -1.0]])
conv.state_dict()['bias'][0] = 0.0
print(conv.state_dict())

# create a dummy tensor to rep an image, the  shape of the image is (1,1,5,5) where:
# (number of inputs, number of channels, number of rows, number of columns)
image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1

# perform convolution
z = conv(image)

# Determining the Size of the Output
K = 2
conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=K)
conv1.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv1.state_dict()['bias'][0] = 0.0
conv1.state_dict()
M = 4
image1 = torch.ones(1, 1, M, M)

# Perform the convolution and verify the size is correct
z1 = conv1(image1)
print("z1:", z1)
print("shape:", z1.shape[2:4])

# Stride parameter

conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)

conv3.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv3.state_dict()['bias'][0] = 0.0
print(conv3.state_dict())
z3 = conv3(image1)
print("z3:", z3)
print("shape:", z3.shape[2:4])

# PADDING
conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=3)
conv4.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv4.state_dict()['bias'][0] = 0.0
conv4.state_dict()
z4 = conv4(image1)
print("z4:", z4)
print("z4:", z4.shape[2:4])

conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=3, padding=1)
conv5.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv5.state_dict()['bias'][0] = 0.0
conv5.state_dict()
z5 = conv5(image1)
print("z5:", z5)
print("z5:", z4.shape[2:4])

# PRACTICE:
Image = torch.randn((1, 1, 4, 4))
# Question 1: Without using the function, determine what the outputs values are as each element:
# Answer:
# As each element of the kernel is zero, and for every  output,
# the image is multiplied  by the kernel, the result is always zero

# Question 2: Use the following convolution object to perform convolution on the tensor Image:
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
conv.state_dict()['weight'][0][0] = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0.0, 0]])
conv.state_dict()['bias'][0] = 0.0
# Answer:
result = conv(Image)

# Question 4: You have an image of size 4. The parameters are as follows kernel_size=2,stride=2.
# What is the size of the output?
# Answer:
img_size = 4
kernel_size = 2
stride = 2
output_size = ((img_size - kernel_size) / stride) + 1  # 2
