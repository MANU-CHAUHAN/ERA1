import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import utils
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
from pytorch_lightning import Callback
from pytorch_lightning.tuner.tuning import Tuner

bn_momentum = 0.3


class LitS10CustomResNet(pl.LightningModule):
    def __init__(self, train_set, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate

        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)
        self.train_dataset = train_set
        self.batch_size = 256
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

        self.final_max = nn.MaxPool2d(4)
        self.fc = nn.Linear(in_features=512, out_features=10, bias=False)

    def forward(self, x):
        """
        The `forward` pass, used during predict/inference in Pytorch Lightning
        """
        prep = self.prep(x)

        layer1 = self.layer1(prep)
        res1 = self.res1(layer1)
        layer1 = layer1 + res1

        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        res3 = self.res3(layer3)
        layer3 = layer3 + res3

        max = self.final_max(layer3)
        out = max.view(max.size(0), -1)

        fc = self.fc(out)

        out = fc.view(-1, 10)

        return out

    def training_step(self, batch, batch_idx):
        """
        Lightning automates the training loop for you and manages all the associated components such as:
        epoch and batch tracking, optimizers and schedulers, and metric reduction. As a user,
        you just need to define how your model behaves with a batch of training data within the training_step().
        When using Lightning, simply override the training_step() method which takes the current batch and the batch_idx
        as arguments.
        """
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        self.train_acc(preds, y)
        self.log(name="train_loss", value=loss, logger=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, logger=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        test_loss = F.cross_entropy(preds, y)
        self.test_acc(preds, y)
        self.log("test_loss", test_loss, logger=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)


class MyCallback(Callback):
    def on_fit_start(self, trainer, pl_module) -> None:
        print("Calling `Fit` now...")

    def on_train_start(self, trainer, pl_module) -> None:
        print("Starting training...")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


train_set, test_set, mean, sdev = utils.get_train_test_datasets(data="cifar10",
                                                                model_name="resnet",
                                                                lr_scheduler="onecycle",
                                                                cutout_prob=0.5)

# data loaders on data sets
train_loader = torch.utils.data.DataLoader(train_set)

test_loader = torch.utils.data.DataLoader(test_set)

tb_logger = TensorBoardLogger(save_dir="../logs/")

model = LitS10CustomResNet(train_set)

# trainer = pl.Trainer(fast_dev_run=10)

trainer = pl.Trainer(auto_lr_find=True,
                     logger=tb_logger,
                     callbacks=[MyCallback()],
                     benchmark=True,
                     enable_checkpointing=True,
                     max_epochs=10,
                     enable_model_summary=True
                     )

tuner = Tuner(trainer)
# tuner.lr_find(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader, min_lr=1e-3, max_lr=10)
# trainer.tune(model)

# Run learning rate finder
lr_finder = trainer.tuner.lr_find(model)

# Results can be found in
print(lr_finder.results)

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()

# update hparams of the model
model.hparams.lr = new_lr

# Fit model
trainer.fit(model, val_dataloaders=test_loader)

# Auto-scale batch size by growing it exponentially (default)
# tuner.scale_batch_size(model, mode="power")

# Fit as normal with new batch size
# trainer.fit(model)
# trainer.test(ckpt_path="best", dataloaders=test_loader)
