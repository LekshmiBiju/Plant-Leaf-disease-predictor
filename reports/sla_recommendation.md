# SLA Recommendation

## Deployment Objective

For plant disease detection, missing a diseased plant can be more
important than producing a false disease alert. Therefore, the
deployment threshold should prioritize disease recall while maintaining
an acceptable level of precision.

## Binary Classification

The four-class prediction problem was additionally evaluated as:

- Healthy = 0
- Any disease = 1

A Precision-Recall curve was generated using the predicted probability
of the image being diseased.

## Operating Threshold

The operating threshold was selected from the Precision-Recall curve
with the target of:

- Recall >= 0.95
- Precision >= 0.80

The selected threshold and its corresponding precision and recall are
stored in:

`models/inference_config.json`

## Deployment Recommendation

For polyhouse deployment, the selected threshold should be used for
healthy-versus-diseased screening. A lower threshold may be preferred
if detecting every possible diseased plant is more important, although
this may increase false-positive alerts.

The threshold should be periodically re-evaluated when new seasonal,
environmental, camera, or disease-stage data becomes available.

## Conclusion

The selected operating threshold provides a practical balance between
detecting diseased leaves and limiting unnecessary disease alerts.
The threshold should be validated on additional field data before
being used as the final production decision threshold.