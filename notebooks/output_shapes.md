Input : (1,3,224,224)

Conv2D(3→32)
Output = (1,32,224,224)

MaxPool2D
Output = (1,32,112,112)

Conv2D(32→64)
Output = (1,64,112,112)

MaxPool2D
Output = (1,64,56,56)

Conv2D(64→128)
Output = (1,128,56,56)

AdaptiveAvgPool2D
Output = (1,128,1,1)