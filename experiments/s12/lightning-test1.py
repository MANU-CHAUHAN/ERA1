import torch
from torch import optim, nn, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

encoder = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 3))

decoder = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 28 * 28))


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)

        self.log("train_loss", loss, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


autoencoder = LitAutoEncoder(encoder, decoder)

train_set = MNIST(root="../data/", download=True, train=True, transform=ToTensor())
test_set = MNIST(root="../data/", download=True, train=False, transform=ToTensor())
train_loader = utils.data.DataLoader(train_set)
test_loader = utils.data.DataLoader(train_set)

tb_logger = TensorBoardLogger(save_dir="../logs/")
trainer = pl.Trainer(limit_train_batches=10, max_epochs=2, logger=tb_logger)

trainer.fit(model=autoencoder, train_dataloaders=train_loader)
trainer.test(ckpt_path="best", dataloaders=test_loader)

encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
