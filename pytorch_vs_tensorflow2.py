# Pytorch basic computational graph
import torch

N, D = 2, 2

x = torch.randn(N, D, requires_grad=True)
y = torch.randn(N, D, requires_grad=True)
z = torch.randn(N, D, requires_grad=True)

print(x)
print(y)
print(z)

a = x * y
print(a)

b = a + z
print(b)

c = torch.sum(b)
print(c)

c.backward()

print(x.grad.data)
print(y.grad.data)
print(z.grad.data)
