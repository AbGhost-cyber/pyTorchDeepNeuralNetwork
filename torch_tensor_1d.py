import torch
# PyTorch Neural Network
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# pip3 install --pre torch torchvision torchaudio

def plotVec(vectors):
    ax = plt.axes()

    # For loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width=0.05, color=vec["color"], head_length=0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.show()


ints_to_tensor = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
new_float_tensor = torch.IntTensor([1, 1, 1, 1])
# convert to another
# print(new_float_tensor.type(torch.FloatTensor))

# get size and dimension
# print(new_float_tensor.size())
# print(new_float_tensor.ndimension())

# reshaping
three_D = new_float_tensor.view(-1, 2)
# print(three_D)

# convert numpy array to a tensor
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
new_tensor = torch.from_numpy(numpy_array)
# print(new_tensor)

# Convert a tensor to a numpy array
back_to_numpy = new_tensor.numpy()
# print("The numpy array from tensor: ", back_to_numpy)
# print("The dtype of numpy array: ", back_to_numpy.dtype)

# Convert a panda series to a tensor
pandas_series = pd.Series([0.1, 2, 0.3, 10.1])
new_tensor = torch.from_numpy(pandas_series.values)

# Using variable to contain the selected index, and pass it to slice operation
selected_indexes = [0, 1]
subset_tensor = new_tensor[selected_indexes]
# print(subset_tensor)

# Calculate the mean for math_tensor
math_tensor = torch.tensor([1.0, -1.0, 1, -1])
# print("The mean is : ", math_tensor.mean())

# Method for calculating the sin result of each element in the tensor
pi_tensor = torch.tensor([0, np.pi / 2, np.pi])
sin = torch.sin(pi_tensor)
# print("The sin result of pi_tensor: ", sin)

len_9_tensor = torch.linspace(-2, 2, steps=9)
# print("Second Try on linspace", len_9_tensor)

# Construct the tensor within 0 to 360 degree
pi_tensor = torch.linspace(0, 2 * np.pi, 100)
pi_tensor.max()
sin_result = torch.sin(pi_tensor)

# Plot sin_result
# plt.plot(pi_tensor.numpy(), sin_result.numpy())
# plt.show()

# tensor addition
# Create two sample tensors
u = torch.tensor([1, 0])
v = torch.tensor([0, 1])
w = u + v
# plotVec([
#     {"vector": u.numpy(), "name": 'u', "color": 'r'},
#     {"vector": v.numpy(), "name": 'v', "color": 'b'},
#     {"vector": w.numpy(), "name": 'w', "color": 'g'}
# ])

# tensor + scalar
u = torch.tensor([1, 2, 3, -1])
v = u + 1
x = 2 * u
if __name__ == '__main__':
    print()
