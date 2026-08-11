# Transfer Learning Concepts

## Feature Extraction

In feature extraction, the pretrained ResNet18 backbone is frozen.
Only the new classification head is trained.

This is useful when:
- The dataset is relatively small.
- The pretrained ImageNet features are already useful.
- Faster training is required.

## Fine-Tuning

In fine-tuning, some pretrained layers are unfrozen and trained with
a small learning rate.

In this project, the `layer4` block is unfrozen after initially
training the classifier.

Fine-tuning allows the pretrained features to adapt to plant leaf
disease characteristics.

## Strategy Used

1. Load ImageNet-pretrained ResNet18.
2. Freeze the backbone.
3. Replace the final fully connected layer with a 4-class classifier.
4. Train the new classifier.
5. Unfreeze `layer4`.
6. Fine-tune using a small learning rate.