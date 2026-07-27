# Why AdaptiveAvgPool2d?

AdaptiveAvgPool2d converts feature maps into a fixed size regardless of the input image size.

Advantages:
- Works with different image sizes.
- Reduces the number of parameters.
- Prevents very large flattened vectors.
- Makes CNN models more flexible.