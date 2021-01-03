import torch

# Tensor concepts
# Shape
# Gradients
# Squeezing and Unsqueezing

t = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=float, requires_grad=True)
v = torch.tensor([9,10,11,12], dtype=float, requires_grad=True)
w = v * t

n = torch.tensor([1,2,3,4])
m1 = n.unsqueeze(1)
m2 = m1.unsqueeze(1)
m3 = m2.squeeze()

print('Scrap Done')