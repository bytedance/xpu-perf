import torch
torch.manual_seed(0)

a = torch.rand(10)
a1 = a.to( device="cuda:3",copy=True, dtype=torch.float32)
print(a1)
b1 = a1.to(device = "cuda:2", copy=True, dtype=torch.float32)
print(b1)


b = torch.rand(10)+1
c = b.to(device="cuda:2", copy=True)

print(c)
d = b.to( device="cuda:3", copy=True)

print(d)

# for i in range(100000000000):
#     d = d*d-d 