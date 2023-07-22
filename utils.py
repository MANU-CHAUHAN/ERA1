import datetime
from typing import Callable, Optional
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import subprocess

import models


# def log_inputs_and_output(func):
#     def wrapper(*args, **kwargs):
#         # Log the function call with its inputs
#         print(f"Function {func.__name__} called with args: {args}, kwargs: {kwargs}")
#
#         # Call the original function and get its return value
#         result = func(*args, **kwargs)
#
#         # Log the return value
#         print(f"Function {func.__name__} returned: {result}")
#
#         return result
#
#     return wrapper


def get_device():
    """
    Returns available torch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""

    print("\n\n⏳ Computing mean and standard deviation...")

    if isinstance(dataset, torchvision.datasets.CIFAR10):
        mean = (np.mean(dataset.data, axis=(0, 1, 2)) / 255).tolist()

        std = (np.std(dataset.data, axis=(0, 1, 2)) / 255).tolist()
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=4)
        mean = torch.zeros(3)
        std = torch.zeros(3)

        for inputs, targets in dataloader:
            for i in range(3):
                mean[i] += inputs[:, i, :, :].mean()
                std[i] += inputs[:, i, :, :].std()

        mean = (mean.div_(len(dataset))).tolist()
        std = (std.div_(len(dataset))).tolist()

    print("\nDone ✅")
    return mean, std


class Cifar10Dataset(torchvision.datasets.CIFAR10):
    """
    Custom Dataset class for CIFAR10 using provided augmentations.
    """

    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def plot_graphs(train_losses, train_accuracy, test_losses, test_accuracy):
    """
    Plot 2 graphs after training and evaluation is done.
    Plots Training vs Test accuracy and Training vs Test loss
    :return: None
    """
    # Plot for accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy, label='Training Accuracy', color='blue')
    plt.plot(test_accuracy, label='Test Accuracy', color='orange')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Plot for loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='green')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def get_optimizer(model, optim_type="adam", lr=0.001, enable_nesterov=False, momentum_value=0.9, weight_decay=5e-4):
    """
    Returns optimizer based in passed values
    :return: Instance of `Optimizer` class but of specific type (from SGD or ADAM)
    """
    if not isinstance(optim_type, str):
        raise NotImplementedError("Type expected for optimizer type is STR")

    optim_type = optim_type.lower()

    if optim_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum_value, nesterov=enable_nesterov)
    elif optim_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"\nRe-check passed arguments to `get_optimizer` in {__file__}.")
    return optimizer


def train_eval_model(model, train_loader, optimizer, device, epochs=1, test=False, test_loader=None,
                     scheduler=None, save_model=False):
    """
    The main `train` and `test` function.
    Takes all components and trains the model for each epoch and runs test code after each train epoch completion.

    :param model: the Pytorch model, subclass of nn.Module to be trained and validated
    :param train_loader: the train dataset loader
    :param optimizer: the optimizer to be used during training
    :param device: the device to use for running the computations (either CPU or CUDA)
    :param epochs: the total number of epochs to be used to run training
    :param test: whether to test model after each epoch of training
    :param test_loader: test data loader to be used if `test` is True
    :param scheduler: Learning Rate Scheduler to be used if provided
    :param save_model: TO save the model or not

    :return: dictionary {'train_losses': [], 'test_losses': [],
                        'train_accuracy': [], 'test_accuracy': []}

    """

    model.train()  # set the train mode

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # iterate over for `epochs` epochs and keep storing valuable info

    for epoch in range(epochs):
        correct = processed = train_loss = 0

        print(f"\n epoch num ================================= {epoch + 1}")

        pbar = tqdm(train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(
                device)  # move data to `device`

            optimizer.zero_grad()  # zero out the gradients to avoid accumulating them over loops

            output = model(data)  # get the model's predictions

            # calculate Negative Log Likelihood loss using ground truth labels and the model's predictions
            loss = F.nll_loss(output, target)

            train_loss += loss.item()  # add up the train loss

            loss.backward()  # The magic function to perform backpropagation and calculate the gradients

            optimizer.step()  # take 1 step for the optimizer and update the weights

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # compare and see how many predictions are correct and then add up the count
            correct += pred.eq(target.view_as(pred)).sum().item()

            processed += len(data)  # total processed data size

        acc = 100 * correct / processed

        train_losses.append(train_loss)

        train_accuracies.append(acc)

        if scheduler:
            print("\n\n\t\t\tLast LR -->", scheduler.get_last_lr())
            scheduler.step()

        pbar.set_description(desc=f'loss={loss.item()} batch_id={batch_idx}')

        train_loss /= len(train_loader.dataset)

        print('\n\t\t\tTrain metrics: accuracy: {}/{} ({:.4f}%)'.format(correct,
                                                                        len(train_loader.dataset),
                                                                        correct * 100 / len(train_loader.dataset)))

        if test:  # moving to evaluation
            model.eval()  # set the correct mode

            correct = test_loss = 0

            with torch.no_grad():  # to disable gradient calculation with no_grad context

                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)

                    output = model(data)

                    # sum up batch loss
                    test_loss += F.nll_loss(output,
                                            target, reduction='sum').item()

                    # get the index of the max log-probability
                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            test_accuracies.append(100. * correct / len(test_loader.dataset))

            print('\n\tTest metrics: average loss: {:.4f}, accuracy: {}/{} ({:.5f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    if save_model:
        torch.save(
            model, f"model-{model}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}.pth")

    return {"train_losses" : train_losses, "test_losses": test_losses, "train_accuracy": train_accuracies,
            "test_accuracy": test_accuracies}


def get_model_name_to_model_object(model_name):
    """
    Returns the model for the corresponding name passed.
    :param model_name: type:str, model name
    :return: the model instance
    """
    device = get_device()
    model_name = model_name.lower()

    if "s9cifar10" in model_name:
        return models.S9CIFAR10().to(device=device)
    elif "s7mnist" in model_name:
        return models.S7MNIST().to(device)
    elif all(x in model_name for x in ["s10", "resnet"]):
        return models.S10CustomResNet().to(device)
    else:
        raise NotImplementedError(
            "\nCheck model names passed, it does not match any of the available models under `models.py` file.")


def get_lr_scheduler(scheduler_name, optimizer, train_loader, total_epochs, max_lr=10, step_size=1, gamma=0.9
                     , pct_start=0.20, anneal_strategy="linear", div_factor=10, final_div_factor=100,
                     enable_three_phase=False, enable_cycle_momentum=True, verbose=0):
    """
    Helper function to get Learning Rate Scheduler.

    Parameters
    ----------
    scheduler_name : the name of the scheduler to use, type: str
    optimizer : the optimizer instance to use
    train_loader : train data loader, applicable where necessary
    max_lr : maximum learning rate to use in case of One Cycle LR Scheduler, applicable where necessary
    step_size : step size to be used to reduce LR in case of Step LR, applicable where necessary
    gamma : gamma value, type: float, between 0 and 1, decides new LR by multiplying the gamma value, applicable where necessary
    total_epochs : total epochs to run the scheduler for, applicable where necessary
    pct_start : The Peak or Max LR percentage of total epochs to use for warm up
    anneal_strategy : the end LR annealing strategy ti be used (`cos` or `linear`)
    div_factor : the division factor to use to get `MAX_LR / div_factor` to get start learning rate, applicable where necessary
    final_div_factor : the final division factor to get end learning rate in cool down phase,`MAX_LR / final_div_factor`, applicable where necessary
    enable_three_phase : type: boolean, flag to enable 3 phase Cycle for LR Scheduler (One Cycle)
    enable_cycle_momentum : type boolean, flag to enable cyclic momentum in scheduler, inverse or increasing LR, applicable where necessary
    verbose : to print out in-built logs for scheduler, if available

    Returns
    -------

    """
    if "steplr" in scheduler_name:
        return lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    elif "onecyc" in scheduler_name:
        return torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                   max_lr=max_lr,
                                                   steps_per_epoch=len(train_loader),
                                                   epochs=total_epochs,
                                                   pct_start=pct_start,
                                                   div_factor=div_factor,
                                                   final_div_factor=final_div_factor,
                                                   anneal_strategy=anneal_strategy,
                                                   three_phase=enable_three_phase,
                                                   cycle_momentum=enable_cycle_momentum,
                                                   verbose=verbose)


def run_lr_finder(model, criterion, start_lr, train_loader, optimizer,
                  optimizer_type="adam",
                  weight_decay=5e-4,
                  num_iterations=300, max_lr=10,
                  log_lr=True, step_mode="exp"):
    """
    Torch Learning Rate finder, helps to run iterations to determine changing loss and plot the graph for Loss & Epochs.
    Helps to decide a good enough LR (usually a high LR to be used in LR Schedulers like One Cycle)

    Parameters
    ----------
    model : the model to be used
    criterion : loss criterion
    start_lr : start learning rate, usually to be set within optimizer
    max_lr : maximum learning rate to be used for running the iterations
    train_loader : the train data loader
    optimizer : optimizer instance
    optimizer_type : optimizer type if `optimizer` is None
    weight_decay : weight decay to be used
    num_iterations : total iterations to run the lr finder
    log_lr : use Log space to plot LR
    step_mode : ste mode (linear or exp)

    Returns None, plots the LR Finder graph
    -------

    """
    from torch_lr_finder import LRFinder

    print(f"\n\nRunning LR finder... 🔍👀 \nStart LR: {start_lr}, End LR: {max_lr}, iterations: {num_iterations},"
          f" step mode: {step_mode}\n")
    
    if not optimizer:
        optimizer = get_optimizer(model=model, optim_type=optimizer_type, lr=start_lr, weight_decay=weight_decay)

    lr_finder = LRFinder(model, optimizer, criterion, device=get_device())
    lr_finder.range_test(train_loader, end_lr=max_lr, num_iter=num_iterations, step_mode=step_mode)
    lr_finder.plot(log_lr=log_lr)
    lr_finder.reset()
    plt.show()


def get_string_to_criterion(cri_str):
    if "crossentropy" in cri_str:
        return nn.CrossEntropyLoss()
    elif "nll" in cri_str:
        return F.nll_loss


def train(model: nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, scheduler: Callable[[Optional[torch.Tensor]], float],
          criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          train_acc: list, train_loss: list, epoch: int):
    """

    Parameters
    ----------
    model : The model instance, subclass of nn.Module
    device : torch.device to be used
    train_loader : DataLoader[Dataset]
    optimizer : torch.optim.Optimizer instance, SGD or Adam
    scheduler : Callable[[Optional[torch.Tensor]], float], the scheduler instance to be used
    criterion : type Callable[[Tensor, Tensor], Tensor], the loss criterion to be used
    train_acc : list to append the training accuracy to
    train_loss : list to append the training loss to
    epoch : Number of the epoch (current epoch)

    Returns None
    -------

    """
    model.train()

    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    acc = 0.0

    # lr_epochs.append((epoch, optimizer.param_groups[0]['lr']))
    print(
        f"\nEpoch num: {epoch}  |  LR: {optimizer.param_groups[0]['lr']:.10f}", end="\n")

    for batch_idx, (data, target) in enumerate(pbar):

        data, target = data.to(device), target.to(device)

        y_pred = model(data)
        loss = criterion(y_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        acc = 100 * correct / processed

        pbar.set_description(desc=f'Loss={loss.item()} Accuracy={acc:.2f}')

    train_loss.append(loss.data.cpu().numpy().item())
    train_acc.append(acc)


def test(model, device, test_loader, test_acc, test_losses):
    """

    Parameters
    ----------
    model : The model instance, subclass of nn.Module
    device : torch.device to be used
    test_loader : DataLoader[Dataset]
    test_acc : list for test accuracy to be added
    test_losses : list for train loss to be added

    Returns None
    -------

    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.5f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc.append(100. * correct / len(test_loader.dataset))


def run_train_and_test(model, device, train_loader, test_loader, optimizer, criterion, scheduler, epochs=1):
    """
    The main Training and Testing Utility function that leverages above `train` and `test` functions.

    `epochs`: type: int, the total number of epochs to run the train and test.

    Also calls `plot_graphs()` to plot train vs test accuracy and train vs test losses
    """

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    print("\n➤ Training started...→")

    for epoch in range(epochs):
        train(model=model,
              device=device,
              train_loader=train_loader,
              optimizer=optimizer,
              criterion=criterion,
              train_acc=train_accuracy,
              train_loss=train_losses,
              epoch=epoch + 1,
              scheduler=scheduler)

        test(model, device, test_loader, test_accuracy, test_losses)

    plot_graphs(train_losses=train_losses, train_accuracy=train_accuracy,
                test_losses=test_losses, test_accuracy=test_accuracy)


def plot_cifar10_aug_images(train_loader, mean, sdev):
    """
    A utility function to plot CIFAR 10 dataset by using passed train loader instance.

    Usually helpful, especially in case of augmentations applied.

    Parameters
    ----------
    train_loader : train data loader instance

    Returns None
    -------

    """

    # channel_means = (0.49196659, 0.48229005, 0.4461573)
    # channel_stdevs = (0.24703223, 0.24348513, 0.26158784)
    def unnormalize(img):
        img = img.numpy().astype(dtype=np.float32)

        for i in range(img.shape[0]):
            img[i] = (img[i] * sdev[i]) + mean[i]

        return np.transpose(img, (1, 2, 0))

    import matplotlib.pyplot as plt
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print(images.shape)
    print(labels.shape)

    num_classes = 10
    # display 10 images from each category.
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    r, c = 10, 11
    n = 5
    fig = plt.figure(figsize=(15, 15))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for i in range(num_classes):
        idx = np.random.choice(np.where(labels[:] == i)[0], n)
        ax = plt.subplot(r, c, i * c + 1)
        ax.text(-1.5, 0.5, class_names[i], fontsize=14)
        plt.axis('off')
        for j in range(1, n + 1):
            plt.subplot(r, c, i * c + j + 1)
            plt.imshow(unnormalize(images[idx[j - 1]]), interpolation='none')
            plt.axis('off')
    plt.show()


def get_train_test_datasets(data, model, cutout_prob=0.5, lr_scheduler=None):
    if "mnist" in data:
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
            root='./data', train=True, download=True, transform=train_transforms)

        test_set = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=test_transforms)

        return train_set, test_set

    elif "cifar10" in data:
        mean, sdev = get_mean_and_std(torchvision.datasets.CIFAR10(root="./data",
                                                                   train=True,
                                                                   download=True,
                                                                   transform=transforms.Compose(
                                                                       [transforms.ToTensor()])))
        if "resnet" in model and "onecyc" in lr_scheduler.lower():
            train_transforms = A.Compose([
                A.Normalize(mean=mean, std=sdev, always_apply=True),
                A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=mean,
                              always_apply=True),
                A.RandomCrop(32, 32, always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=mean, p=cutout_prob),
                ToTensorV2()
            ])

            test_transforms = A.Compose([
                A.Normalize(mean=mean, std=sdev,
                            always_apply=True),
                A.HorizontalFlip(p=0.5),
                ToTensorV2()
            ])
        # else:
        #     train_transforms = A.Compose([
        #         A.HorizontalFlip(p=0.2),
        #         A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
        #                            rotate_limit=30, p=0.20),
        #         A.CoarseDropout(max_holes=1, p=0.15, max_height=16,
        #                         max_width=16, min_holes=1, min_height=16,
        #                         min_width=16, fill_value=mean),
        #         # A.MedianBlur(blur_limit=3, p=0.1),
        #         A.HueSaturationValue(p=0.1),
        #         #   A.GaussianBlur(blur_limit=3, p=0.12),
        #         # A.RandomBrightnessContrast(brightness_limit=0.09,contrast_limit=0.1, p=0.15),
        #         A.Normalize(mean=mean, std=sdev),
        #         ToTensorV2()
        #     ])
        #
        #     test_transforms = A.Compose([
        #         A.Normalize(mean=mean, std=sdev),
        #         ToTensorV2()
        #     ])

        train_set = Cifar10Dataset(
            train=True, download=True, transform=train_transforms)

        test_set = Cifar10Dataset(
            train=False, download=True, transform=test_transforms)

        return train_set, test_set, mean, sdev


def check_requirements(requirements_file='requirements.txt'):
    try:
        # Use the 'pip check' command to check if all requirements are already installed
        result = subprocess.run(['pip', 'check', '-r', requirements_file], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        return result.returncode == 0  # All requirements are already installed if returncode is 0
    except FileNotFoundError:
        return False  # pip command not found


def install_requirements(requirements_file='requirements.txt'):
    if check_requirements(requirements_file):
        print("All requirements already installed.")
        return

    try:
        # Use the 'pip install' command to install the packages from the requirements file
        result = subprocess.run(['pip', 'install', '-r', requirements_file], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("All requirements installed successfully.")
        else:
            print("An error occurred while installing requirements:")
            print(result.stderr)
    except FileNotFoundError:
        print("Error: 'pip' command not found. Please ensure you have Python and pip installed.")


def fraction_to_float(value):
    if isinstance(value, str):
        if '/' in value:
            numerator, denominator = map(float, value.split('/'))
            return numerator / denominator
        else:
            return float(value)
    else:
        return float(value)


if __name__ == "__main__":
    pass
    # mean, sdev = get_mean_and_std(torchvision.datasets.CIFAR10(root="./data", train=True,
    #                                                            download=True,
    #                                                            transform=transforms.Compose([transforms.ToTensor()])))
    #
    # print(mean, sdev)
