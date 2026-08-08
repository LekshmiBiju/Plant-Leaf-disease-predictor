import torch
from torch import nn, optim
from tqdm import tqdm

from model import LeafDiseaseCNN
from dataset import train_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeafDiseaseCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


for epoch in range(5):
    loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")


torch.save(model.state_dict(), "models/leaf_cnn_epoch5.pth")

print("Training Complete!")