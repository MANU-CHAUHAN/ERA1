import torch.nn as nn
import pytorch_lightning as pl

from model import (Transformer,
                   Encoder,
                   Decoder,
                   EncoderBlock,
                   DecoderBlock,
                   FeedForwardBlock,
                   InputEmbeddings,
                   PositionalEncoding,
                   ProjectionLayer,
                   MultiHeadAttentionBlock)


def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int,
                      N: int = 6,
                      h: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)

    # create positional encoding layers next
    src_pos = PositionalEncoding(
        d_model=d_model, seq_len=src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(
        d_model=d_model, seq_len=tgt_seq_len, dropout=dropout)

    # create list to hold Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        # create or intialize sub-layers for current Encoder block
        encoder_self_attention = MultiHeadAttentionBlock(
            d_model=d_model, h=h, dropout=dropout)
        feed_forward_block = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, dropout=dropout)

        # create 1 instance of Encoder block
        encoder_block = EncoderBlock(self_attention_block=encoder_self_attention,
                                     feed_forward_block=feed_forward_block,
                                     dropout=dropout)

        # add this 1 block to Encoder block list
        encoder_blocks.append(encoder_block)

    # create a list to hold Decoder blocks
    decoder_blocks = []
    for _ in range(N):  # have `N` total blocks
        # initalize sub-layers for each block
        decoder_self_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, h=h, dropout=dropout)

        decoder_cross_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, h=h, dropout=dropout)

        feed_forward_block = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, dropout=dropout)

        # create 1 decoder block now
        decoder_block = DecoderBlock(self_attention_block=decoder_self_attention_block,
                                     cross_attention_block=decoder_cross_attention_block,
                                     feed_forward_block=feed_forward_block,
                                     dropout=dropout)

        # add this 1 decoder block to decoder blocks list
        decoder_blocks.append(decoder_block)

    # link encoders
    encoder = Encoder(layers=nn.ModuleList(encoder_blocks))

    # link decoders
    decoder = Decoder(layers=nn.ModuleList(decoder_blocks))

    # create a projection layer for decoder side
    projection_layer = ProjectionLayer(
        d_model=d_model, vocab_size=tgt_vocab_size)

    # create the Transformer
    transformer = Transformer(encoder=encoder,
                              decoder=decoder,
                              src_embed=src_embed, tgt_embed=tgt_embed,
                              src_pos=src_pos, tgt_pos=tgt_pos,
                              projection_layer=projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
