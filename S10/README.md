# Residual Connections in CNNs and One Cycle Policy

![](https://miro.medium.com/v2/resize:fit:1200/1*6hF97Upuqg_LdsqWY6n_wg.png)

![](https://miro.medium.com/v2/resize:fit:952/1*_7CeyMZaolYgmfgbl85dqg.jpeg)

----
----

Deep convolutional neural networks have led to a series of breakthroughs for image classification tasks.

Challenges such as ILSVRC and COCO saw people exploiting deeper and deeper models to achieve better results. Clearly network depth is of crucial importance.

Due to the difficult nature of real world tasks or problems being thrown at deep neural networks, the size of the networks is bound to increase when one wants to attain high levels of accuracy on deep learning tasks. The reason being that the network needs to extract high number of critical patterns or features from the training data and learn very complex yet meaning full representations, from their combinations, at later layers in the network, from raw input data like high resolution multi colored images. Without considerable depth, the network will not be able to combine intricate small/mid/high level features in much more complex ways in order to actually LEARN the inherent complex nature of the problem being solved with raw data.
Hence, the first solution to solve complex problems was to make your neural networks deep, really Deeeeeeeeeeeeeeeeeeeeeeeeeep. For experiments and research purposes, the depth of some networks were set to be around more than 100s of layers in order to get high understanding and training accuracy for the problem at hand.

So, the authors of Deep Residual Networks paper asked one very important but neglected question: Is learning better networks as easy as stacking more layers ?

Trying this, however, proved highly inefficient without giving the expected performance gains. The reason, you ask !
In theory, as the number of layers in a plain neural network increases, then it should provide us increasingly complex representations, resulting better learning and thus higher accuracies. 

Contrary to the belief, with experiments to prove, with increase in number of layers, the network’s train accuracy began to drop. See the image below.

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*4DzKazEW_cpKCby4GRYogg.png)
*Theory VS Practical training errors when number of layers increases in a PLAIN NEURAL NETWORK*


From the paper: When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error.
So, to address the degradation problem, the authors introduced a deep residual learning framework, as part of a hypothesis hoping to address the issue.
The idea is that instead of letting layers learn the underlying mapping, let the network fit the residual mapping. So, instead of say H(x), initial mapping, let the network fit, F(x) := H(x)-x which gives H(x) := F(x) + x
The approach is to add a shortcut or a skip connection that allows info to flow, well just say, more easily from one layer to the next’s next layer, ie you bypass data along with normal CNN flow from one layer to the next layer after the immediate next.
A residual block:

How does this help?
As hypothesised by the authors: adding skip connection not only allows data to flow between layers easily , it also allows the learning of identity function very trivial, thus allowing the same information to flow without transformations. The network is able to learn better when having identity mapping by it’s side to use, as learning perturbations with reference to identity mappings are easier than to learn function as a new one from scratch.

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*6WlIo8W1_Qc01hjWdZy-1Q.png)


#### Result:
![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*YEG_mgaWF5RJs4fg3n9XPA.png)

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*AmOha3nT2shU64Po0BE8Nw.png)

[Source](https://medium.com/@manu1992/what-are-deep-residual-networks-or-why-resnets-are-important-40a94b562d81)

[One Cycle Policy Pytorch](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#:~:text=The%201cycle%20policy%20anneals%20the,than%20the%20initial%20learning%20rate.)


### Tasks:
1. Write a custom ResNet architecture for CIFAR10 that has the following architecture:

#### PrepLayer
- Conv 3x3 (stride 1, padding 1)
- Batch Normalization (BN)
- ReLU Activation
- 64k

#### Layer 1
X = Conv 3x3 (stride 1, padding 1) >> MaxPool2D >> BN >> ReLU
- 128k

R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU) )(X)
- 128k

Add(X, R1)

#### Layer 2
- Conv 3x3
- MaxPooling2D
- BN
- ReLU
- 256k

#### Layer 3
X = Conv 3x3 (stride 1, padding 1) >> MaxPool2D >> BN >> ReLU
- 512k

R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU) )(X)
- Output Size: 512k

Add(X, R2)

MaxPooling with Kernel Size 4

#### FC Layer
- SoftMax

#### Additionally
- Uses One Cycle Policy such that:
  - Total Epochs = 24
  - Max at Epoch = 5
  - LRMIN = FIND
  - LRMAX = FIND
  - NO Annihilation
- Uses this transform: RandomCrop 32x32 (after padding of 4) >> FlipLR >> Followed by CutOut(8x8)

#### Training Configuration
- Batch size = 512
- Optimizer: ADAM
- Loss function: CrossEntropyLoss

#### Target Accuracy: 90%

## Model:
![Image Alt Text](../resources/S10_model.png)

| Layer (type)   | Output Shape      | Param #    |
|----------------|-------------------|------------|
| Conv2d-1       | [-1, 64, 32, 32]  | 1,728      |
| BatchNorm2d-2  | [-1, 64, 32, 32]  | 128        |
| ReLU-3         | [-1, 64, 32, 32]  | 0          |
| Conv2d-4       | [-1, 128, 32, 32] | 73,728     |
| MaxPool2d-5    | [-1, 128, 16, 16] | 0          |
| BatchNorm2d-6  | [-1, 128, 16, 16] | 256        |
| ReLU-7         | [-1, 128, 16, 16] | 0          |
| Conv2d-8       | [-1, 128, 16, 16] | 147,456    |
| BatchNorm2d-9  | [-1, 128, 16, 16] | 256        |
| ReLU-10        | [-1, 128, 16, 16] | 0          |
| Conv2d-11      | [-1, 128, 16, 16] | 147,456    |
| BatchNorm2d-12 | [-1, 128, 16, 16] | 256        |
| ReLU-13        | [-1, 128, 16, 16] | 0          |
| Conv2d-14      | [-1, 256, 16, 16] | 294,912    |
| MaxPool2d-15   | [-1, 256, 8, 8]   | 0          |
| BatchNorm2d-16 | [-1, 256, 8, 8]   | 512        |
| ReLU-17        | [-1, 256, 8, 8]   | 0          |
| Conv2d-18      | [-1, 512, 8, 8]   | 1,179,648  |
| MaxPool2d-19   | [-1, 512, 4, 4]   | 0          |
| BatchNorm2d-20 | [-1, 512, 4, 4]   | 1,024      |
| ReLU-21        | [-1, 512, 4, 4]   | 0          |
| Conv2d-22      | [-1, 512, 4, 4]   | 2,359,296  |
| BatchNorm2d-23 | [-1, 512, 4, 4]   | 1,024      |
| ReLU-24        | [-1, 512, 4, 4]   | 0          |
| Conv2d-25      | [-1, 512, 4, 4]   | 2,359,296  |
| BatchNorm2d-26 | [-1, 512, 4, 4]   | 1,024      |
| ReLU-27        | [-1, 512, 4, 4]   | 0          |
| MaxPool2d-28   | [-1, 512, 1, 1]   | 0          |
| Linear-29      | [-1, 10]          | 5,120      |

Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0

Input size (MB): 0.01
Forward/backward pass size (MB): 6.44
Params size (MB): 25.07
Estimated Total Size (MB): 31.53



#### Notebooks and musings:

1. Tried with the given architecture straight forward and the augmnetations.

2. LR Finder turned out to be very helpful, however the stepping in log space seems to better cover LRs than linear stepping.

3. The plot and suggested LR is used to pick the Max LR for the OneCycleLR and `div_factor=10` while `final_div_factor=100`, experimented with 1000 too.

4. There are 3 Notebooks that cross 90% test accuracy, these are usual Jupyter Notebooks that have code and plots together.

5. The `direct` notebook clones repo and runs the command line tool to find lr and then the suggessted LR is used to OneCycle Policy with all required arguments passed (file name: `S10_direct_in_colab.ipynb`).

6. The augmentations used clearly helped to smooth out (and in some cases widen) the loss curve with iterations during LR finder too.

7. Small value of ColorJitter was tried and that helped a little but soon a decent value was excessive as network failed to cross 90-92% for training phase.

8. DID NOT USE **0.175** to multiply with output of FC layer.
