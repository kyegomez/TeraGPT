import torch
from teragpt.main import TeraGPT

model = TeraGPT(
    dim=4096,
    depth=6,
    heads=8,
    num_tokens=20000,
)

x = torch.randint(0, 20000, (1, 4096))

out = model(x)
print(out.shape)
