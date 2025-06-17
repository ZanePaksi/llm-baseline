import torch

x = torch.rand(5, 3)
print(x)

y = torch.cuda.is_available()
print(y)