import argparse
from functools import partial

import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import utils

cv = partial(nn.Conv2d, bias=False)
bn = nn.BatchNorm2d
relu = nn.ReLU

# optimize operations if available (https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/24)
torch.backends.cudnn.benchmark = True

arg_parser = argparse.ArgumentParser(
    description="Training program for various models with multiple options.")

arg_parser.add_argument('--lr', default=0.001, type=float,
                        help="Learning rate to set for the model")

arg_parser.add_argument('--dataset', default=None,
                        type=str, help="The dataset to use.")

arg_parser.add_argument('--model', default=None, type=str,
                        help="The model to use for training.")

arg_parser.add_argument('--epochs', default=1, type=int,
                        help="Number of epochs to run the training for.")

arg_parser.add_argument('--lr_scheduler', default="steplr", type=str,
                        help="Type of Learning Rate scheduler to use.\nAvailable 1. `steplr` 2. `cycliclr`")

arg_parser.add_argument('--gamma', default=0.9,
                        help="The `gamma` value to use for LR scheduler")

arg_parser.add_argument('--step_size', default=1, type=int,
                        help="Step Size for LR scheduler to change LR = LR * gamma after step size.")

arg_parser.add_argument('--optim', default="sgd", type=str,
                        help="Optimizer to select (sgd or adam")

arg_parser.add_argument('--save', default=False, type=bool,
                        help="To save the model or not after training.")

arg_parser.add_argument('--max-lr', default=1.23e-03, type=float,
                        help="The maximum LR to be with One Cycle LR Scheduler.")

arg_parser.add_argument('--batch', default=32, type=int,
                        help="\nThe batch size for dataloader.")
arg_parser.add_argument('--pct_start', default=0.20, type=float,
                        help="the end of warm up phase and peak or max LR epoch float value out of total epochs")
arg_parser.add_argument('--anneal_fn', type=str, help="Annealing function to use (Linear or Cosine)")

arg_parser.add_argument('--help', action='help', default=argparse.SUPPRESS,
                        help='Available arguments\n'
                             '1. `--lr` for learning rate, default 0.01\n'
                             '2. `--dataset` for selecting the dataset, available options MNIST and CIFAR10\n'
                             '3. `--model` model name, check models.py\n'
                             '4. `--epochs` for setting number of epochs, default 1.\n'
                             '5. `--lr_scheduler` for selecting which learning rate scheduler to use\n'
                             '6. `--gamma` for gamma value to be used between 0 and 1.0\n'
                             '7. `--step_size` for number of steps after which to change LR in scheduler\n'
                             '8. `--optim` type of optimizer to use: 1.SGD 2.Adam\n'
                             '9. `--save` if to save the model or not (true or false)\n'
                             '10. `--max-lr` if cyclic policy is used, the maximum learning rate to be used.\n'
                             '11. `--pct_start` : the end of warm up phase and peak or max LR epoch float value out of total epochs\n'
                             '12. `--anneal_fn` : annealing function to decrease LR in cool down phase of LR scheduler')

args = arg_parser.parse_args()

model = utils.get_model_name_to_model_object(model_name=args.model.lower())

optimizer = utils.get_optimizer(model=model, optim_type=args.optim, lr=args.lr)

epochs = args.epochs

pct_start = args.pct_start

anneal_fn = args.anneal_fn

dataset = args.dataset.lower()

BATCH = args.batch

device = utils.get_device()

dataloader_args = dict(shuffle=True, batch_size=BATCH, num_workers=2, pin_memory=True)

if dataset == "mnist":
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-6.9, 6.9), fill=(1,)),
        # translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomAffine(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # Test data transformations

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=train_transforms)

    test_set = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=test_transforms)

elif args.dataset.lower() == "cifar10":
    mean, sdev = utils.get_mean_and_std(torchvision.datasets.CIFAR10(root="./data",
                                                                     train=True,
                                                                     download=True,
                                                                     transform=transforms.Compose(
                                                                         [transforms.ToTensor()])))
    if "custom_resnet" in model and "one_cycle" in lr_scheduler:
        train_transforms = A.Compose([
            A.Normalize(mean=mean, std=sdev, always_apply=True),
            A.PadIfNeeded(min_height=40, min_width=40,
                          border_mode=cv2.BORDER_CONSTANT, value=mean, always_apply=True),
            A.RandomCrop(32, 32, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.Cutout(num_holes=1, max_h_size=8,
                     max_w_size=8, fill_value=mean, p=1),
            ToTensorV2()
        ])

        test_transforms = A.Compose([
            A.Normalize(mean=mean, std=sdev,
                        always_apply=True),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ])
    else:
        train_transforms = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                               rotate_limit=30, p=0.20),
            A.CoarseDropout(max_holes=1, p=0.15, max_height=16,
                            max_width=16, min_holes=1, min_height=16,
                            min_width=16, fill_value=mean),
            # A.MedianBlur(blur_limit=3, p=0.1),
            A.HueSaturationValue(p=0.1),
            #   A.GaussianBlur(blur_limit=3, p=0.12),
            # A.RandomBrightnessContrast(brightness_limit=0.09,contrast_limit=0.1, p=0.15),
            A.Normalize(mean=mean, std=sdev),
            ToTensor()
        ])

        test_transforms = A.Compose([
            A.Normalize(mean=mean, std=sdev),
            ToTensor()
        ])

    train_set = utils.Cifar10Dataset(
        train=True, download=True, transform=train_transforms)

    test_set = utils.Cifar10Dataset(
        train=False, download=True, transform=test_transforms)

# data loaders on data sets
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, **dataloader_args)

test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

lr_scheduler = utils.get_lr_scheduler(scheduler_name=args.scheduler,
                                      optimizer=optimizer,
                                      step_size=args.step_size,
                                      gamma=args.gamma, total_epochs=epochs, pct_start=pct_start,
                                      anneal_strategy=anneal_fn,
                                      train_loader=train_loader)

utils.run_train_and_test(model=model, device=device, train_loader=train_loader, test_loader=test_loader,
                         optimizer=optimizer, criterion=nn.CrossEntropyLoss(), scheduler=lr_scheduler, epochs=epochs)

# results = utils.train_eval_model(False, model, train_loader, optimizer, device, epochs=epochs, test=True,
#                                  test_loader=test_loader, scheduler=lr_scheduler)
#
# utils.plot_graphs(train_losses=results["train_losses"], test_losses=results["test_losses"],
#                   train_accuracy=results["train_accuracies"], test_accuracy=results["test_accuracies"])
