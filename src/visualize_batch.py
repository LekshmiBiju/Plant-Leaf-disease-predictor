import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import LeafDiseaseDataset, transform

# Create dataset
train_ds = LeafDiseaseDataset("data/train", transform=transform)

# Create dataloader
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

# Get one batch
images, labels = next(iter(train_loader))

# Undo normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

img = images[0] * std + mean
img = img.permute(1, 2, 0)

plt.imshow(img)
plt.title(f"Label: {labels[0].item()}")
plt.axis("off")
plt.savefig("sample_batch.png")
plt.show()