import torch
import torch.nn as nn
import matplotlib.pylab as plt
import numpy as np

# Objective: To deal with several problems associated with optimization and see how momentum can improve our results.

torch.manual_seed(0)


# This function will plot a cubic function and the parameter values obtained via Gradient Descent.
def plot_cubic(w, optimizer):
    LOSS = []
    # parameter values
    W = torch.arange(-4, 4, 0.1)
    # plot the loss function
    for w.state_dict()['linear.weight'][0] in W:
        LOSS.append(cubic(w(torch.tensor([[1.0]]))).item())
    w.state_dict()['linear.weight'][0] = 4.0
    n_epochs = 10
    parameter = []
    loss_list = []

    # n_epochs
    # Use PyTorch custom module to implement a ploynomial function
    for n in range(n_epochs):
        optimizer.zero_grad()
        loss = cubic(w(torch.tensor([[1.0]])))
        loss_list.append(loss.detach())
        parameter.append(w.state_dict()['linear.weight'][0].detach().data.item())
        loss.backward()
        optimizer.step()
    plt.plot(parameter, loss_list, 'ro', label='parameter values')
    plt.plot(W.numpy(), LOSS, label='objective function')
    plt.xlabel('w')
    plt.ylabel('l(w)')
    plt.legend()


# This function will plot a 4th order function and the parameter values obtained via Gradient Descent.
# You can also add Gaussian noise with a standard deviation determined by the parameter std
def plot_fourth_order(w, optimizer, std=0, color='r', paramlabel='parameter values', objfun=True):
    W = torch.arange(-4, 6, 0.1)
    LOSS = []
    for w.state_dict()['linear.weight'][0] in W:
        LOSS.append(fourth_order(w(torch.tensor([[1.0]]))).item())
    w.state_dict()['linear.weight'][0] = 6
    n_epochs = 100
    parameter = []
    loss_list = []

    # n_epochs
    for n in range(n_epochs):
        optimizer.zero_grad()
        loss = fourth_order(w(torch.tensor([[1.0]]))) + std * torch.randn(1, 1)
        loss_list.append(loss.detach())
        parameter.append(w.state_dict()['linear.weight'][0].detach().data.item())
        loss.backward()
        optimizer.step()

    # Plotting
    if objfun:
        plt.plot(W.numpy(), LOSS, label='objective function')
    plt.plot(parameter, loss_list, 'ro', label=paramlabel, color=color)
    plt.xlabel('w')
    plt.ylabel('l(w)')
    plt.legend()


# Create a linear model

class one_param(nn.Module):
    # Constructor
    def __init__(self, input_size, output_size):
        super(one_param, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    # Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat


# Create a one_param object
w = one_param(1, 1)


# Define a function to output a cubic with saddle points

def cubic(yhat):
    out = yhat ** 3
    return out


# Create an optimizer without momentum
optimizer = torch.optim.SGD(w.parameters(), lr=0.01, momentum=0)

# Plot the model
plot_cubic(w, optimizer)

# Create an optimizer with momentum
optimizer = torch.optim.SGD(w.parameters(), lr=0.01, momentum=0.9)
# Plot the model
plot_cubic(w, optimizer)


# Create a function to calculate the fourth order polynomial
# In this section, we will create a fourth order polynomial with a local minimum at 4 and a global minimum a -2.
# We will then see how the momentum parameter affects convergence to a global minimum.
# The fourth order polynomial is given by
def fourth_order(yhat):
    out = torch.mean(2 * (yhat ** 4) - 9 * (yhat ** 3) - 21 * (yhat ** 2) + 88 * yhat + 48)
    return out


# Make the prediction without momentum
optimizer = torch.optim.SGD(w.parameters(), lr=0.001)
plot_fourth_order(w, optimizer)

# Make the prediction with momentum
optimizer = torch.optim.SGD(w.parameters(), lr=0.001, momentum=0.9)
plot_fourth_order(w, optimizer)

# In this section, we will create a fourth order polynomial with a local minimum at 4 and
# a global minimum a -2, but we will add noise to the function when the Gradient is calculated.
# We will then see how the momentum parameter affects convergence to a global minimum.
# with no momentum, we get stuck in a local minimum

# Make the prediction without momentum when there is noise
optimizer = torch.optim.SGD(w.parameters(), lr=0.001)
plot_fourth_order(w, optimizer, std=10)

# Make the prediction with momentum when there is noise
optimizer = torch.optim.SGD(w.parameters(), lr=0.001, momentum=0.9)
plot_fourth_order(w, optimizer, std=10)

# PRACTICE:
# Create two  SGD objects with a learning rate of  0.001. Use the default momentum parameter value
# for one and a value of  0.9 for the second. Use the function plot_fourth_order with an std=100,
# to plot the different steps of each. Make sure you run the function on two independent cells.

optimizer1 = torch.optim.SGD(w.parameters(), lr=0.001)
plot_fourth_order(w, optimizer1, std=100, color='black', paramlabel='parameter values with optimizer 1')
optimizer2 = torch.optim.SGD(w.parameters(), lr=0.001, momentum=0.9)
plot_fourth_order(w, optimizer2, std=100, color='red', paramlabel='parameter values with optimizer 2', objfun=False)
