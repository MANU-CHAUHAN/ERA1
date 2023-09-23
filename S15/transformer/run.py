from config import get_config
from train import train_model

cfg = get_config()
cfg['batch_size'] = 6
cfg['preload'] = None
cfg['num_epochs'] = 10
train_model(config=cfg)
