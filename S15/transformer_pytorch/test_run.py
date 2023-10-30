from config import get_config
import torch

cfg = get_config()

cfg['batch_size'] = 2
cfg['preload'] = None
cfg['num_epochs'] = 3

from train import train_model

train_model(config=cfg)
