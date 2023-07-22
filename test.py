import utils
import torch
import torch.nn as nn

if __name__ == "__main__":
    utils.install_requirements()
    model = "s10resnet"

    optimizer = "adam"

    epochs = 24

    pct_start = 4 / 24

    anneal_fn = "linear"

    dataset = "cifar10"

    BATCH = 32

    start_lr = 1e-03

    max_lr = 1.27e-03

    device = utils.get_device()

    lr_scheduler = "onecycle"

    dataloader_args = dict(shuffle=True, batch_size=BATCH, num_workers=1, pin_memory=False)

    train_set, test_set, mean, sdev = utils.get_train_test_datasets(data=dataset, model=model, lr_scheduler=lr_scheduler)

    # data loaders on data sets
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, **dataloader_args)

    test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

    model = utils.get_model_name_to_model_object(model_name=model)

    optimizer = utils.get_optimizer(model=model, optim_type=optimizer)

    # utils.run_lr_finder(model=model, criterion=nn.CrossEntropyLoss(), start_lr=start_lr, max_lr=12,
    #                     train_loader=train_loader, optimizer=optimizer, num_iterations=500)

    lr_scheduler = utils.get_lr_scheduler(scheduler_name=lr_scheduler,
                                          optimizer=optimizer,
                                          total_epochs=epochs, pct_start=pct_start,
                                          anneal_strategy=anneal_fn,
                                          train_loader=train_loader, max_lr=max_lr)

    utils.run_train_and_test(model=model, device=device, train_loader=train_loader, test_loader=test_loader,
                             optimizer=optimizer, criterion=utils.get_string_to_criterion("crossentropy"),
                             scheduler=lr_scheduler, epochs=epochs)
