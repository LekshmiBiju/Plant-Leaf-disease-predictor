# Task 6 Classification Report

## Model

`models/resnet18_day12_best.pth`

## Test Set

Test images: 400

## Precision, Recall and F1-Score

```text
              precision    recall  f1-score   support

     healthy     1.0000    1.0000    1.0000       100
early_blight     0.9804    1.0000    0.9901       100
 late_blight     1.0000    0.9900    0.9950       100
   leaf_mold     1.0000    0.9900    0.9950       100

    accuracy                         0.9950       400
   macro avg     0.9951    0.9950    0.9950       400
weighted avg     0.9951    0.9950    0.9950       400
```

## Confusion Matrix

```text
[[100   0   0   0]
 [  0 100   0   0]
 [  0   1  99   0]
 [  0   1   0  99]]
```
