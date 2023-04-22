import torch
from torch.nn import Linear, Module

# Define w = 2 and b = -1 for y = wx + b
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)


def forward(x):
    yhat = w * x + b
    return yhat


# Predict y = 2x - 1 at x = 1
x1 = torch.tensor([[1.0]])
yhat = forward(x1)
print("The prediction: ", yhat)

# Create x Tensor and check the shape of x tensor

x1 = torch.tensor([[1.0], [2.0]])
print("The shape of x: ", x1.shape)

# we can use the linear class as well for the prediction
torch.manual_seed(1)

lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))

# state_dict() Returns a Python dictionary object corresponding to the layers of each parameter tensor.
print("Python dictionary: ", lr.state_dict())
print("keys: ", lr.state_dict().keys())
print("values: ", lr.state_dict().values())

print("weight:", lr.weight)
print("bias:", lr.bias)

x1 = torch.tensor([[1.0], [2.0]])
yhat = lr(x1)
print("The prediction: ", yhat)


# Customize Linear Regression Class

class LR(Module):

    def __init__(self, in_size, out_size):
        super(LR, self).__init__()
        self.linear = Linear(in_features=in_size, out_features=out_size)

    def forward(self, x):
        out = self.linear(x)
        return out


lr = LR(in_size=1, out_size=1)
print("The parameters: ", list(lr.parameters()))
print("Linear model: ", lr.linear)

x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)

if __name__ == '__main__':
    print()