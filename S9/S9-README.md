### The tasks were:

##### Write a new network:

1. has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
2. total RF must be more than 44
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. use albumentation library and apply:
   1. horizontal flip
   2. shiftScaleRotate
   3. coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

#### The approach:

1. Started with basic network but keeping in mind the dimensions of the dataset (CIFAR10) hence minimal kernels from beginning.
2. To keep network efficient and small (less 3x3 later), added 1x1 in betweens to allow mixing of data. Also wanted to keep params as low as possible.
3. Added `dilated` convolution
4. Used `strides` > 1
5. Added `depthwise` convolution in block 4
6. Used Cyclic Learning Rate scheduler for `first 100 epochs`, to allow for frequent movements in loss space and avoid local minima.
7. After a lot of experiments, `base_lr` of 0.003 and `max_lr` of 0.1 was used in Cyclic LR, with `step_up` and `step_down` as 10 and `gamma`=0.85. **Note:** `batch size = 512`
8. `Augmentations` used are:
   1. HorizontalFlip
   2. ShiftScaleRotate
   3. Cutout (max hole=1 and size = 16x16)
   4. HueSaturationValue
9. All the above allowed to touch `70% test accuracy` on `39th epoch` and `train accuracy` of `65.66%`.
10. After `first 100 epochs` (SGD optimizer):
    1. Train accuracy: 77.29 %
    2. Test accuracy: 78.02 % (average test loss = 0.6390)
11. For next 150 epochs used Step_LR, moved to Adam and base lr of 0.005 to shake things, up and restart (resumed) learning, final 87% train and 84.03% test accuracies.
12. For next 150 epochs moved back to SGD (🤦) with Step LR, `step_size=2`, `gamma=0.93`, lr=0.001, momentum=0.9.
13. **Model architecture:**
    | Layer (type)         | Output Shape     | Param # |
    | -------------------- | ---------------- | ------- |
    | Conv2d-1             | [-1, 16, 32, 32] | 432     |
    | ReLU-2               | [-1, 16, 32, 32] | 0       |
    | BatchNorm2d-3        | [-1, 16, 32, 32] | 32      |
    | Conv2d-4             | [-1, 16, 32, 32] | 256     |
    | ReLU-5               | [-1, 16, 32, 32] | 0       |
    | BatchNorm2d-6        | [-1, 16, 32, 32] | 32      |
    | Conv2d-7             | [-1, 32, 32, 32] | 4,608   |
    | ReLU-8               | [-1, 32, 32, 32] | 0       |
    | BatchNorm2d-9        | [-1, 32, 32, 32] | 64      |
    | Conv2d-10            | [-1, 32, 16, 16] | 9,216   |
    | ReLU-11              | [-1, 32, 16, 16] | 0       |
    | BatchNorm2d-12       | [-1, 32, 16, 16] | 64      |
    | Conv2d-13            | [-1, 64, 16, 16] | 576     |
    | ReLU-14              | [-1, 64, 16, 16] | 0       |
    | BatchNorm2d-15       | [-1, 64, 16, 16] | 128     |
    | Conv2d-16            | [-1, 32, 16, 16] | 2,048   |
    | ReLU-17              | [-1, 32, 16, 16] | 0       |
    | BatchNorm2d-18       | [-1, 32, 16, 16] | 64      |
    | Conv2d-19            | [-1, 64, 16, 16] | 1,152   |
    | ReLU-20              | [-1, 64, 16, 16] | 0       |
    | BatchNorm2d-21       | [-1, 64, 16, 16] | 128     |
    | Conv2d-22            | [-1, 48, 16, 16] | 3,072   |
    | ReLU-23              | [-1, 48, 16, 16] | 0       |
    | BatchNorm2d-24       | [-1, 48, 16, 16] | 96      |
    | Conv2d-25            | [-1, 64, 8, 8]   | 1,728   |
    | ReLU-26              | [-1, 64, 8, 8]   | 0       |
    | BatchNorm2d-27       | [-1, 64, 8, 8]   | 128     |
    | Conv2d-28            | [-1, 32, 8, 8]   | 2,048   |
    | ReLU-29              | [-1, 32, 8, 8]   | 0       |
    | BatchNorm2d-30       | [-1, 32, 8, 8]   | 64      |
    | Conv2d-31            | [-1, 64, 8, 8]   | 576     |
    | ReLU-32              | [-1, 64, 8, 8]   | 0       |
    | BatchNorm2d-33       | [-1, 64, 8, 8]   | 128     |
    | Conv2d-34            | [-1, 128, 8, 8]  | 512     |
    | ReLU-35              | [-1, 128, 8, 8]  | 0       |
    | BatchNorm2d-36       | [-1, 128, 8, 8]  | 256     |
    | Conv2d-37            | [-1, 64, 8, 8]   | 8,192   |
    | ReLU-38              | [-1, 64, 8, 8]   | 0       |
    | BatchNorm2d-39       | [-1, 64, 8, 8]   | 128     |
    | Conv2d-40            | [-1, 96, 8, 8]   | 1,728   |
    | ReLU-41              | [-1, 96, 8, 8]   | 0       |
    | BatchNorm2d-42       | [-1, 96, 8, 8]   | 192     |
    | Conv2d-43            | [-1, 64, 8, 8]   | 6,144   |
    | ReLU-44              | [-1, 64, 8, 8]   | 0       |
    | BatchNorm2d-45       | [-1, 64, 8, 8]   | 128     |
    | Conv2d-46            | [-1, 64, 5, 5]   | 36,864  |
    | ReLU-47              | [-1, 64, 5, 5]   | 0       |
    | BatchNorm2d-48       | [-1, 64, 5, 5]   | 128     |
    | Conv2d-49            | [-1, 96, 5, 5]   | 1,728   |
    | BatchNorm2d-50       | [-1, 96, 5, 5]   | 192     |
    | ReLU-51              | [-1, 96, 5, 5]   | 0       |
    | Conv2d-52            | [-1, 64, 5, 5]   | 6,144   |
    | BatchNorm2d-53       | [-1, 64, 5, 5]   | 128     |
    | ReLU-54              | [-1, 64, 5, 5]   | 0       |
    | Conv2d-55            | [-1, 64, 5, 5]   | 576     |
    | Conv2d-56            | [-1, 32, 5, 5]   | 2,048   |
    | BatchNorm2d-57       | [-1, 32, 5, 5]   | 64      |
    | ReLU-58              | [-1, 32, 5, 5]   | 0       |
    | Conv2d-59            | [-1, 48, 3, 3]   | 1,728   |
    | ReLU-60              | [-1, 48, 3, 3]   | 0       |
    | BatchNorm2d-61       | [-1, 48, 3, 3]   | 96      |
    | Conv2d-62            | [-1, 10, 3, 3]   | 480     |
    | ReLU-63              | [-1, 10, 3, 3]   | 0       |
    | BatchNorm2d-64       | [-1, 10, 3, 3]   | 20      |
    | AdaptiveAvgPool2d-65 | [-1, 10, 1, 1]   | 0       |
|  |    |        |
|-------------------|--------------------|---------|
| **Total params:**  |     **94,116**     |         |
| Trainable params: |     94,116         |         |
| Non-trainable params: |     0             |         |
|-------------------|--------------------|---------|
| Input size (MB):  |      0.01          |         |
| Forward/backward pass size (MB): |  3.83 |         |
|   Params size (MB):  |      0.36         |         |
| **Estimated Total Size (MB):** |  **4.20** |         |

​    