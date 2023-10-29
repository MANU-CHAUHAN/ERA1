import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_all_sentences(ds, lang):
    """
    Get all sentences in the dataset, for the given language
    """
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]

        # get source and target texts
        src_txt = src_target_pair['translation'][self.src_lang]
        tgt_txt = src_target_pair['translation'][self.tgt_lang]

        # transform texts into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_txt).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_txt).ids

        # add SOS, EOS and PAD to each sentence
        # calculate number of pads needed for encoder and decoder separately
        # eg: 350(max seq_len) - 5(input seq or sentence len) - 2(sos and eos) = 343
        enc_num_padding_tokens = self.seq_len - \
                                 len(enc_input_tokens) - 2  # consider SOS and EOS
        # consider SOS only as EOS is in the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # make sure the padded sequence is not negative, seq_len should be calculated separately beforehand for max value
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")

        # Add <s> and </s> token and <pad>, with calculation for number of <pad>s needed
        # Encoder structure is: <SOS>, <input_token_ids>, <EOS>, <PAD>, ... <PAD>
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             enc_num_padding_tokens, dtype=torch.int64)
            ], dim=0
        )

        # Decoder: add only <SOS>, <input_token_ids> and <PAD>...<PAD> , as <EOS> is to be predicted by the decoder
        # <SOS>, <input_token_ids>, <PAD>, ... <PAD>
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64)
            ], dim=0
        )

        # for label, add only <EOS> token with input and pads
        # <decoder input token ids>, <EOS>, <PAD>, ... <PAD>
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64)

            ]
        )

        # checks for dimensions
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # seq_len
            "decoder_input": decoder_input,  # seq_len
            # (1, 1, seq_len)
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_len) & (1, seq_len, seq_len)
            "label"        : label,  # (seq_len)
            "src_text"     : src_txt,
            "tgt_text"     : tgt_txt
        }


def causal_mask(size):
    # mask with upper triangle, with an extra size=1 removed above the diagonal, specified by diagonal=1
    """
    This is commonly used in sequence-to-sequence models like transformers to ensure that the decoder only attends to positions
      that have been previously decoded, preventing it from looking into the future positions.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class MyDataset(pl.LightningDataModule):
    def __init__(self, config,
                 ds_name='opus_books',
                 val_batch=1):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.max_len_src = None
        self.max_len_tgt = None
        self.tokenizer_src = None
        self.tokenizer_tgt = None
        self.ds_name = ds_name
        self.config = config
        self.train_batch_size = config['batch_size']
        self.val_batch_size = val_batch

    def prepare_data(self):
        """download first time, single process called first, even in case of multi GPU setup"""
        ds_raw = load_dataset(self.ds_name,
                              f"{self.config['lang_src']}-{self.config['lang_tgt']}",
                              split='train')

        # build tokenizers first time, then re-load in setup()
        get_or_build_tokenizer(config=self.config,
                               ds=ds_raw,
                               lang=self.config['lang_src'])

        get_or_build_tokenizer(config=self.config,
                               ds=ds_raw,
                               lang=self.config['lang_tgt'])

    def setup(self, stage=None):
        """ assign already downloaded dataset
            in case of multi-gpu setup, this method is called per GPU"""

        ds_raw = load_dataset(self.ds_name,
                              f"{self.config['lang_src']}-{self.config['lang_tgt']}",
                              split='train')

        # load already built tokenizers
        self.tokenizer_src = get_or_build_tokenizer(config=self.config,
                                                    ds=ds_raw,
                                                    lang=self.config['lang_src'])

        self.tokenizer_tgt = get_or_build_tokenizer(config=self.config,
                                                    ds=ds_raw,
                                                    lang=self.config['lang_tgt'])

        # keep 90% data for train and 10% for val
        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size

        train_ds_raw, val_ds_raw = random_split(
            ds_raw, [train_ds_size, val_ds_size])

        if stage in ["fit", None]:
            self.train_ds = BilingualDataset(ds=train_ds_raw,
                                             tokenizer_src=self.tokenizer_src,
                                             tokenizer_tgt=self.tokenizer_tgt,
                                             src_lang=self.config['lang_src'],
                                             tgt_lang=self.config['lang_tgt'],
                                             seq_len=self.config['seq_len'])
        if stage in ["val", None]:
            self.val_ds = BilingualDataset(ds=val_ds_raw,
                                           tokenizer_src=self.tokenizer_src,
                                           tokenizer_tgt=self.tokenizer_tgt,
                                           src_lang=self.config['lang_src'],
                                           tgt_lang=self.config['lang_tgt'],
                                           seq_len=self.config['seq_len'])

        # find max len of sentence in source and target languages
        self.max_len_src = self.max_len_tgt = 0

        for item in ds_raw:
            src = item['translation'][self.config['lang_src']]
            tgt = item['translation'][self.config['lang_tgt']]
            src_ids = self.tokenizer_src.encode(src).ids
            tgt_ids = self.tokenizer_tgt.encode(tgt).ids
            self.max_len_src = max(self.max_len_src, len(src_ids))
            self.max_len_tgt = max(self.max_len_tgt, len(tgt_ids))

        print(f"Max len of source sentence: {self.max_len_src}")
        print(f"Max len of target sentence: {self.max_len_tgt}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=True)
