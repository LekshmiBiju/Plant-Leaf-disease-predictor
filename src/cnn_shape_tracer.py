import torch
import torch.nn as nn

def trace_shapes(x, layers):
    for layer in layers:
        x = layer(x)
        print(f"{layer.__class__.__name__:15} -> {tuple(x.shape)}")

x = torch.randn(1, 3, 224, 224)

layers = [
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1)
]

trace_shapes(x, layers)