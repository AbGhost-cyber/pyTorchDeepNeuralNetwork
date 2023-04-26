import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

# Objective: To study Convolution and review how the different operations change the
# relationship between input and output.

# Multiple Output Channels
# Pytorch randomly assigns values to each kernel. However, use kernels that have been developed to detect edges:
conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
Gx = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0.0, -1.0]])
Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

conv1.state_dict()['weight'][0][0] = Gx
conv1.state_dict()['weight'][1][0] = Gy
conv1.state_dict()['weight'][2][0] = torch.ones(3, 3)

# Each kernel has its own bias, so set them all to zero:
conv1.state_dict()['bias'][:] = torch.tensor([0.0, 0.0, 0.0])
print(conv1.state_dict()['bias'])
# Print out each kernel
for x in conv1.state_dict()['weight']:
    print(x)

# Create an input image to represent the input X
image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1

# Plot it as an image
plt.imshow(image[0, 0, :, :].numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.colorbar()
plt.show()

# Perform convolution using each channel:
out = conv1(image)
print(out.shape)

# Print out each channel as a tensor or an image
for channel, image in enumerate(out[0]):
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

image1 = torch.zeros(1, 1, 5, 5)
image1[0, 0, 2, :] = 1
print(image1)
plt.imshow(image1[0, 0, :, :].detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.show()

# In this case, the second channel fluctuates, and the first and the third channels produce a constant value.
out1 = conv1(image1)
for channel, image in enumerate(out1[0]):
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

# Multiple Input Channels
image2 = torch.zeros(1, 2, 5, 5)
image2[0, 0, 2, :] = -2
image2[0, 1, 2, :] = 1
print(image2)

# plot out each image
for channel, image in enumerate(image2[0]):
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3)
Gx1 = torch.tensor([[0.0, 0.0, 0.0], [0, 1.0, 0], [0.0, 0.0, 0.0]])
conv3.state_dict()['weight'][0][0] = 1 * Gx1
conv3.state_dict()['weight'][0][1] = -2 * Gx1
conv3.state_dict()['bias'][:] = torch.tensor([0.0])
print(conv3.state_dict()['weight'])

# perform the convolution
conv3(image2)

# Multiple Input and Output Channels
conv4 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3)
conv4.state_dict()['weight'][0][0] = torch.tensor([[0.0, 0.0, 0.0], [0, 0.5, 0], [0.0, 0.0, 0.0]])
conv4.state_dict()['weight'][0][1] = torch.tensor([[0.0, 0.0, 0.0], [0, 0.5, 0], [0.0, 0.0, 0.0]])

conv4.state_dict()['weight'][1][0] = torch.tensor([[0.0, 0.0, 0.0], [0, 1, 0], [0.0, 0.0, 0.0]])
conv4.state_dict()['weight'][1][1] = torch.tensor([[0.0, 0.0, 0.0], [0, -1, 0], [0.0, 0.0, 0.0]])

conv4.state_dict()['weight'][2][0] = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0.0, -1.0]])
conv4.state_dict()['weight'][2][1] = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

# For each output, there is a bias, so set them all to zero:
conv4.state_dict()['bias'][:] = torch.tensor([0.0, 0.0, 0.0])

# Create a two-channel image and plot the results:
image4 = torch.zeros(1, 2, 5, 5)
image4[0][0] = torch.ones(5, 5)
image4[0][1][2][2] = 1
for channel, image in enumerate(image4[0]):
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

# perform the convolution
z = conv4(image4)
