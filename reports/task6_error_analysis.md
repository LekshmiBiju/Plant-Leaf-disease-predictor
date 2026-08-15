# Task 6 — Error Analysis

## Model

The best ResNet18 model was evaluated on the held-out Day 12 test set.

Model:

`models/resnet18_day12_best.pth`

Test set:

`data/day12/test`

The test set contains 400 images.

## Overall Performance

The model correctly classified 398 out of 400 test images.

- Correct predictions: 398
- Incorrect predictions: 2
- Test accuracy: 99.50%

## Common Confusions

The confusion matrix shows two classification errors.

### Late Blight → Early Blight

One late blight image was incorrectly classified as early blight.

These two disease classes can have visually similar symptoms, particularly when lesions or discoloration appear at similar stages.

### Leaf Mold → Early Blight

One leaf mold image was incorrectly classified as early blight.

The visual similarity between disease symptoms can make classification difficult in some borderline samples.

## Error Analysis

Only two genuine misclassified samples were found in the 400-image held-out test set. Both errors were incorrectly predicted as early blight.

The errors were inspected individually using the files stored in:

`reports/errors/`

The low number of errors indicates that the ResNet18 model performs strongly on the test dataset.

## Misclassification Count

| Error Type | Count |
|---|---:|
| Late Blight → Early Blight | 1 |
| Leaf Mold → Early Blight | 1 |
| Total | 2 |

## Note on 5+ Error Requirement

The task specification requests five or more misclassified examples. However, the evaluated test set contains only two actual misclassified images.

Therefore, only the two genuine errors are included. Additional images were not artificially labeled as misclassified because doing so would produce incorrect evaluation results.