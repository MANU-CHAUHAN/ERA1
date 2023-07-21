import argparse
from functools import partial

import torch
import torch.nn as nn

import utils

cv = partial(nn.Conv2d, bias=False)
bn = nn.BatchNorm2d
relu = nn.ReLU

# optimize operations if available (https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/24)
torch.backends.cudnn.benchmark = True


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


arg_parser = argparse.ArgumentParser(
    description="Training Deep Learning program for various models with multiple options.",
    formatter_class=CustomFormatter
)

arg_parser.add_argument(
    '--lr', default=0.001, type=float,
    help="Learning rate to set for the model"
)

arg_parser.add_argument(
    '--dataset', default=None,
    type=str, help="The dataset to use."
)

arg_parser.add_argument(
    '--model', default=None, type=str,
    help="The model to use for training."
)

arg_parser.add_argument(
    '--epochs', default=1, type=int,
    help="Number of epochs to run the training for."
)

arg_parser.add_argument(
    '--lr-scheduler', default="steplr", type=str,
    help="Type of Learning Rate scheduler to use.\nAvailable options: 1. `StepLR` 2. `OneCycleLR`"
)

arg_parser.add_argument(
    '--gamma', default=0.9,
    help="The `gamma` value to use for the LR scheduler."
)

arg_parser.add_argument(
    '--step_size', default=1, type=int,
    help="Step Size for the LR scheduler to change LR = LR * gamma after the step size."
)

arg_parser.add_argument(
    '--optim', default="sgd", type=str,
    help="Optimizer to select (sgd or adam)"
)

arg_parser.add_argument(
    '--save', default=False, type=bool,
    help="To save the model or not after training."
)

arg_parser.add_argument(
    '--max-lr', default=1.23e-03, type=float,
    help="The maximum LR to be used with the One Cycle LR Scheduler."
)

arg_parser.add_argument(
    '--batch', default=32, type=int,
    help="The batch size for the dataloader."
)

arg_parser.add_argument(
    '--pct_start', default=0.20, type=float,
    help="The end of the warm-up phase and the peak or max LR epoch as a float value out of the total epochs."
)

arg_parser.add_argument(
    '--anneal_fn', type=str,
    help="Annealing function to use (Linear or Cosine)"
)

arg_parser.add_argument(
    '--Help', action='help', default=argparse.SUPPRESS,
    help='''Available arguments:

    1. `--lr`: Learning rate, default 0.001

    2. `--dataset`: Selecting the dataset, available options MNIST and CIFAR10

    3. `--model`: Model name, check models.py

    4. `--epochs`: Setting the number of epochs, default 1.

    5. `--lr-scheduler`: Selecting which learning rate scheduler to use
       Available options: 1. `steplr` 2. `cycliclr`

    6. `--gamma`: Gamma value to be used between 0 and 1.0

    7. `--step_size`: Number of steps after which to change LR in the scheduler

    8. `--optim`: Type of optimizer to use: 1. SGD 2. Adam

    9. `--save`: If to save the model or not (true or false)

    10. `--max-lr`: If cyclic policy is used, the maximum learning rate to be used.

    11. `--batch`: The batch size for the dataloader.

    12. `--pct_start`: The end of the warm-up phase and peak or max LR epoch as a float value out of total epochs.

    13. `--anneal_fn`: Annealing function to decrease LR in the cool-down phase of the LR scheduler'''
)

args = arg_parser.parse_args()

model = args.model.lower()

optimizer = args.optim.lower()

epochs = args.epochs

pct_start = args.pct_start

anneal_fn = args.anneal_fn

dataset = args.dataset.lower()

BATCH = args.batch

device = utils.get_device()

lr_scheduler = args.lr_scheduler.lower()

dataloader_args = dict(shuffle=True, batch_size=BATCH, num_workers=2, pin_memory=True)

train_set, test_set = utils.get_train_test_datasets(data=dataset, model=model, lr_scheduler=lr_scheduler)

# data loaders on data sets
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, **dataloader_args)

test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

model = utils.get_model_name_to_model_object(model_name=model)

optimizer = utils.get_optimizer(model=model, optim_type=optimizer, lr=args.lr)

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
