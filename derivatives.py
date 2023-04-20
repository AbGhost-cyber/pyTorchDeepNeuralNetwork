import torch
import matplotlib.pylab as plt

# setting grad to True means that the tensor will track all
# operations that are performed on it, with this we can
# compute gradients of some other variable with respect to x
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
# take the derivative
y.backward()
# print(x.grad)
# print('data:', x.data)
# print('grad_fn:', x.grad_fn)
# print('grad:', x.grad)
# print("is_leaf:", x.is_leaf)
# print("requires_grad:", x.requires_grad)

# Calculate the y = x^2 + 2x + 1, then find the derivative

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
# print("The result of y = x^2 + 2x + 1: ", y)
y.backward()


# print("The derivative at x = 2: ", x.grad)


class SQ(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        result = i ** 2
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        i, = ctx.saved_tensors
        grad_output = 2 * i
        return grad_output


x = torch.tensor(2.0, requires_grad=True)
sq = SQ.apply

y = sq(x)
y
# print(y.grad_fn)
y.backward()
# print(x.grad)

# Calculate f(u, v) = v * u + u^2 at u = 1, v = 2
u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
f = u * v + u ** 2
# print("The result of v * u + u^2: ", f)
f.backward()
# print("The partial derivative with respect to u: ", u.grad)
# print("The partial derivative with respect to u: ", v.grad)

# Calculate the derivative with multiple values
x = torch.linspace(-10, 10, 10, requires_grad=True)
Y = x ** 2
y = torch.sum(x ** 2)

y.backward()

plt.plot(x.detach().numpy(), Y.detach().numpy(), label='function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='derivative')
plt.xlabel('x')
plt.legend()
plt.show()

# Take the derivative of Relu with respect to multiple value. Plot out the function and its derivative
x = torch.linspace(-10, 10, 1000, requires_grad=True)
Y = torch.relu(x)
y = Y.sum()
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label='function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='derivative')
plt.xlabel('x')
plt.legend()
plt.show()

# Practice: Calculate the derivative of f = u * v + (u * v) ** 2 at u = 2, v = 1
u = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(1.0, requires_grad=True)
f = u * v + (u * v) ** 2
f.backward()
print(u.grad)

if __name__ == '__main__':
    print()
