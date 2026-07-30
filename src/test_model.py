import torch
from model import LeafDiseaseCNN

model = LeafDiseaseCNN(num_classes=4)

x = torch.randn(8, 3, 224, 224)

output = model(x)

print("Output Shape:", output.shape)
print("Parameters:", sum(p.numel() for p in model.parameters()))

assert output.shape == (8, 4)

print("All tests passed!")