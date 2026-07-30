# Plant Leaf Disease Predictor

## Python Version
Python 3.13.12

## PyTorch
Installed successfully.

## GPU Status
-CUDA Available: False
-Device: CPU
- This system does not have an NVIDIA CUDA-enabled GPU, so PyTorch is running on the CPU.


## PyTorch Verification

torch.cuda.is_available(): False

Device: CPU

## Dataset Summary

Early Blight: 1000 images

Healthy: 1000 images

Late Blight: 999 images

Leaf Mold: 1000 images

## Batch Information

Batch Shape:
Images: (8, 3, 224, 224)

Labels: (8)

A sample batch visualization has been generated and saved as sample_batch.png.

## Folder Structure
- data/raw
- data/processed
- models
- notebooks
- src