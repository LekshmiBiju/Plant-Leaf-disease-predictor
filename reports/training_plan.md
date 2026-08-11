# ResNet18 Training Plan

## Stage 1 - Feature Extraction

- Load ImageNet-pretrained ResNet18.
- Freeze the complete backbone.
- Replace the final fully connected layer.
- Use 4 output classes.
- Train only the new classifier.

## Stage 2 - Fine-Tuning

- Unfreeze the `layer4` block.
- Keep earlier ResNet18 layers frozen.
- Use a smaller learning rate.
- Continue training on the plant leaf dataset.
- Monitor validation accuracy and validation loss.

## Model Selection

The model with the best validation performance will be saved as:

models/resnet18_best.pth