# Data Augmentation Notes

## Training Augmentations

The training pipeline uses the following transformations:

- Resize to 256 × 256 pixels
- RandomResizedCrop to 224 × 224 pixels
- RandomHorizontalFlip
- RandomRotation up to ±15 degrees
- ColorJitter for brightness, contrast, and saturation
- Conversion to PyTorch tensor
- ImageNet normalization

## Validation Transformations

Validation images are not randomly augmented. They are only:

- Resized to 224 × 224 pixels
- Converted to tensors
- Normalized using ImageNet mean and standard deviation

## Label-Safe Transformations

The selected transformations are considered label-safe for plant leaf disease classification because they change the viewing conditions without changing the disease category.

Horizontal flipping, small rotations, moderate crops, and moderate brightness/contrast/saturation changes preserve the disease characteristics visible on the leaf.

Large rotations, extreme crops, or transformations that remove disease symptoms should be avoided because they may make the image unrealistic or remove important disease features.

## Reason for Augmentation

Data augmentation increases the diversity of training examples and helps the CNN generalize to leaves captured under different camera angles, distances, and lighting conditions. It can also reduce overfitting to the training dataset.