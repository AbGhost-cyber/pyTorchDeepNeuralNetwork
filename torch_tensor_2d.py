import torch
# PyTorch Neural Network
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Convert 2D List to 2D Tensor

twoD_list = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
twoD_tensor = torch.tensor(twoD_list)
# print("The New 2D Tensor: ", twoD_tensor)
# print("The dimension of twoD_tensor: ", twoD_tensor.ndimension())
# print("The shape of twoD_tensor: ", twoD_tensor.shape)
# print("The shape of twoD_tensor: ", twoD_tensor.size())
# print("The number of elements in twoD_tensor: ", twoD_tensor.numel())

# try converting to numpy and back
twoD_numpy = twoD_tensor.numpy()
new_twoD_tensor = torch.from_numpy(twoD_numpy)

X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]])
X_times_Y = X * Y
# print(X_times_Y)

# Practice: try to convert Pandas Series to tensor
df = pd.DataFrame({'A': [11, 33, 22], 'B': [3, 3, 2]})
numpy_df = torch.from_numpy(df.values)

# Use tensor_obj[row, column] and tensor_obj[row][column] to access certain position
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
# print("What is the value on 2nd-row 3rd-column? ", tensor_example[1, 2])
# print("What is the value on 2nd-row 3rd-column? ", tensor_example[1][2])

# matrix
A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B = torch.mm(A, B)
print("The result of A * B: ", A_times_B)

if __name__ == '__main__':
    print(xx)
