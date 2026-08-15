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

## Day 8 — Class Imbalance Handling

The class distribution of the training dataset was analyzed and
saved in `reports/class_balance.csv`.

To reduce the effect of class imbalance, a
`WeightedRandomSampler` was used during CNN training. This gives
higher sampling probability to classes with fewer training samples.

### Before vs After Balancing

| Metric | Before Balancing | After Balancing |
|---|---:|---:|
| Validation Accuracy | 87.25% | 91.50% |

The baseline validation accuracy before balancing was 87.25%.
The value after balancing was obtained from the Day 8 training run.

Per-class recall was also calculated on the validation dataset and
saved in `reports/per_class_recall.txt`.

### Evaluation

The purpose of comparing the two runs is to determine whether
weighted sampling improves the performance of underrepresented
classes, particularly their recall, rather than relying only on
overall validation accuracy.

## Task 4 — ResNet18 Transfer Learning

A pretrained ResNet18 model was fine-tuned for the four plant leaf
disease classes.

### Transfer Learning Strategy

1. The pretrained ResNet18 backbone was initially frozen.
2. The final fully connected layer was replaced with a 4-class classifier.
3. The classifier was trained while the backbone remained frozen.
4. The `layer4` block was then unfrozen for fine-tuning.
5. A smaller learning rate was used during fine-tuning.
6. The best model weights were saved as `models/resnet18_best.pth`.

### Metrics Comparison

| Model | Validation Accuracy |
|---|---:|
| Task 3 CNN Baseline | 87.25% |
| ResNet18 Transfer Learning | 97.38% |

ResNet18 achieved 97.38% validation accuracy, exceeding the
baseline CNN accuracy of 87.25% by 10.13 percentage points.

### Training Details

- Device: CPU
- Training images: 3200
- Validation images: 800
- Best validation accuracy: 97.38%
- Best model: `models/resnet18_best.pth`

## ResNet18 Fine-Tuning

A pretrained ImageNet ResNet18 model was fine-tuned for the four
plant leaf disease classes.

Training used a two-phase strategy:

1. The pretrained backbone was frozen and the new classifier head
   was trained using Adam with learning rate 1e-3.
2. The `layer4` block was unfrozen and fine-tuned together with the
   classifier. A smaller learning rate of 1e-5 was used for `layer4`.

Baseline CNN validation accuracy: 87.25%.

Best ResNet18 validation accuracy: 97.50%.

Hardware: CPU.

Training time: 78.01 minutes.

Best checkpoint:
`models/resnet18_leaf_best.pth`

Class mapping:
`models/class_names.json`

## ResNet18 vs MobileNetV2

| Model | Validation Accuracy | Model Size | CPU Latency |
|---|---:|---:|---:|
| ResNet18 | 97.38% | 42.72 MB MB | 18.40 ms/image |
| MobileNetV2 | 95.00% | 8.74 MB | 38.47 ms/image |

## Edge vs Server Deployment

ResNet18 provides higher classification accuracy but has a larger model
and higher computational cost. MobileNetV2 is designed for resource-
constrained devices and provides a smaller model with lower CPU inference
latency. Therefore, MobileNetV2 is more suitable for edge deployment on
camera nodes or low-power devices, while ResNet18 is more suitable for
server-side deployment where computational resources are available and
maximum accuracy is preferred.

## Confusion Matrix & Classification Metrics

### Objective

Evaluate the best ResNet18 model on a completely held-out test set and analyze its classification performance using a confusion matrix, per-class accuracy, and misclassification analysis.

### Test Set

The Day 12 evaluation was performed on the held-out test set located at:

`data/day12/test`

The test set contains 400 images across four leaf disease classes:

- Healthy
- Early Blight
- Late Blight
- Leaf Mold

The test set was not used during model training or validation, preventing training-data leakage.

### Model

The best Day 12 ResNet18 checkpoint was used:

`models/resnet18_day12_best.pth`

The model was evaluated on CPU.

### Test Results

| Metric | Result |
|---|---:|
| Test Images | 400 |
| Correct Predictions | 398 |
| Incorrect Predictions | 2 |
| Test Accuracy | 99.50% |

### Per-Class Accuracy

| Class | Accuracy |
|---|---:|
| Healthy | 100.00% |
| Early Blight | 100.00% |
| Late Blight | 99.00% |
| Leaf Mold | 99.00% |

### Confusion Matrix

The confusion matrix was generated using the held-out test set and saved as:

`reports/confusion_matrix_day12.png`

The matrix showed:


              Predicted
             Healthy  Early  Late  Mold

Healthy         100      0     0     0
Early             0    100     0     0
Late              0      1    99     0
Mold              0      1     0    99

### Results

- Test images: 400
- Correct predictions: 398
- Incorrect predictions: 2
- Test accuracy: 99.50%

### Error Analysis

Two genuine misclassified samples were identified:

1. Late Blight → Early Blight
2. Leaf Mold → Early Blight

The misclassified samples are stored in:

`reports/errors/`

A misclassification gallery was also generated and saved as:

`reports/day12_misclassification_gallery.png`

Only two genuine errors were found because the model correctly classified
398 of the 400 test images. Additional samples were not artificially
classified as errors.