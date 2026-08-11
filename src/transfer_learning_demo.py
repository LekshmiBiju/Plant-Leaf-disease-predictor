import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------
# 1. DEVICE
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ---------------------------------------------------------
# 2. LOAD PRETRAINED RESNET18
# ---------------------------------------------------------

weights = models.ResNet18_Weights.IMAGENET1K_V1

backbone = models.resnet18(weights=weights)

print("\nPretrained ResNet18 loaded successfully.")


# ---------------------------------------------------------
# 3. PRINT MODEL HIERARCHY
# ---------------------------------------------------------

print("\nResNet18 Module Hierarchy:")
print(backbone)


# ---------------------------------------------------------
# 4. FREEZE BACKBONE
# ---------------------------------------------------------

for param in backbone.parameters():
    param.requires_grad = False


# ---------------------------------------------------------
# 5. REPLACE CLASSIFIER
# ---------------------------------------------------------

num_classes = 4

in_features = backbone.fc.in_features

backbone.fc = nn.Linear(
    in_features,
    num_classes
)


# The new classifier must be trainable
for param in backbone.fc.parameters():
    param.requires_grad = True


backbone = backbone.to(device)


# ---------------------------------------------------------
# 6. COUNT PARAMETERS
# ---------------------------------------------------------

trainable = sum(
    p.numel()
    for p in backbone.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in backbone.parameters()
)


print("\nParameter Count:")
print("Trainable parameters:", trainable)
print("Total parameters:", total)


# ---------------------------------------------------------
# 7. FORWARD PASS WITH SAMPLE LEAF TENSOR
# ---------------------------------------------------------

dummy_input = torch.randn(
    1, 3, 224, 224
).to(device)

with torch.no_grad():
    output = backbone(dummy_input)


print("\nForward Pass:")
print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)


# ---------------------------------------------------------
# 8. CHECK OUTPUT CLASSES
# ---------------------------------------------------------

print("\nNumber of output classes:", output.shape[1])


# ---------------------------------------------------------
# 9. SAVE REPORT
# ---------------------------------------------------------

with open(
    "reports/transfer_learning_demo.txt",
    "w"
) as f:

    f.write("Transfer Learning Demo - Day 9\n")
    f.write("================================\n\n")

    f.write(f"Device: {device}\n")
    f.write("Model: ResNet18\n")
    f.write("Weights: ImageNet pretrained\n\n")

    f.write("Strategy:\n")
    f.write("- Freeze pretrained backbone\n")
    f.write("- Replace final classifier\n")
    f.write("- Train new classifier\n")
    f.write("- Unfreeze layer4 later\n")
    f.write("- Fine-tune with a small learning rate\n\n")

    f.write(f"Total parameters: {total}\n")
    f.write(f"Trainable parameters: {trainable}\n\n")

    f.write(
        f"Input shape: {tuple(dummy_input.shape)}\n"
    )

    f.write(
        f"Output shape: {tuple(output.shape)}\n"
    )

    f.write(
        f"Number of classes: {output.shape[1]}\n"
    )


print(
    "\nReport saved to "
    "reports/transfer_learning_demo.txt"
)

print("\nDay 9 demo completed successfully.")