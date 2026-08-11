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

## Validation and Early Stopping

The CNN was trained using an 80/20 train-validation split.

- Training images: 3200
- Validation images: 800
- Device: CPU
- Early stopping patience: 3 epochs
- Epochs completed: 31
- Best validation accuracy: 0.8725
- Best model: `models/leaf_cnn_best.pth`
- Training curves: `reports/training_curves.png`

## Data Augmentation

Data augmentation was introduced using torchvision.transforms.

Training images were augmented using random resized cropping,
horizontal flipping, rotation, and moderate color changes.

Validation images were not randomly augmented so that validation
performance remained consistent.

The augmentation visualization is available at:

`reports/augmentation_samples.png`

The training and validation results are available at:

`reports/training_curves.png`

## Label-Safe Data Augmentation

The following transformations were used for training:

- RandomHorizontalFlip: Label-safe because flipping a leaf horizontally does not change the disease category.
- RandomRotation: Small rotations are label-safe because the disease symptoms remain unchanged when the leaf is viewed at a different orientation.
- RandomResizedCrop: Label-safe when the crop still contains sufficient leaf and disease information.
- ColorJitter: Small changes in brightness, contrast, and saturation are label-safe because they simulate different lighting conditions while preserving disease patterns.

Validation data uses only deterministic resizing and normalization to provide a consistent evaluation.

These transformations are appropriate for plant leaf disease images because they change the appearance or orientation of the image without changing the underlying disease label.