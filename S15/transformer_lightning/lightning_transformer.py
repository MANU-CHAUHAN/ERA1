import argparse
import os
import random

import utils
import dill as pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler

from model import Transformer


class TransformerPL(pl.LightningModule):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 src_seq_len: int,
                 tgt_seq_len: int,
                 d_model: int = 512,
                 N: int = 6,
                 h: int = 8,
                 dropout: float = 0.1,
                 d_ff: int = 2048):
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        self.model = utils.build_transformer(src_vocab_size=self.src_vocab_size,
                                             tgt_vocab_size=self.tgt_vocab_size,
                                             src_seq_len=self.src_seq_len,
                                             tgt_seq_len=self.tgt_seq_len,
                                             d_model=self.d_model,
                                             N=self.N,
                                             h=self.h,
                                             dropout=self.dropout,
                                             d_ff=self.d_ff)
