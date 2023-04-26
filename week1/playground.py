import torch


def gen_linspace(start, end, n):
    # first get the step size
    step_size = (end - start) / (n - 1)
    steps = []
    print("step size is: ", step_size)
    # steps must include start
    steps.append(start)
    # keep track of the last inserted item
    last_inserted = start
    for count in range(n - 1):
        next_step = round(last_inserted + step_size, 2)
        last_inserted = next_step
        steps.append(next_step)

    return steps


if __name__ == '__main__':
    # x = torch.tensor(1.0, requires_grad=True)
    # y = x ** 2
    # y.backward()
    # print(x.detach())
    # print(x.grad)
    # print(x.grad.data)
    im3 = torch.zeros(1, 2, 5, 5)
    print(im3.size())
    # print(gen_linspace(start=0, end=2, n=9))
    # print(gen_linspace(-3, 3, 0.1))
