import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import (ResidualConnection,
                     LayerNormalization,
                     InputEmbeddings,
                     PositionalEncoding,
                     FeedForwardBlock,
                     MultiHeadAttentionBlock)


class EncoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # residual connection's forward expects `x` and `sublayer`
        # the first level of residual connection of q, k, v for encoder's self attention is processed by these two lines
        # here q, k, v all are same (x) [Encoder]
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        # next we send this processed block to feed forward, which has residual connection too
        #
        # original x has been updated already, now x refers to incoming data from previous block of self-attention in Encoder

        # now sublayer is a feed forward block
        x = self.residual_connection[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # this is self-attention in Decoder, the first attention block in architecture diagram
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # this is the cross attention block, getting query from itself while K and V comes from encoded output from Encoder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))

        # Feed Forward block as the thrid block
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # this is the final projection layer on top of decoder and `vocab_size` is output vocab size
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)  # convert source seq to embedding
        src = self.src_pos(src)  # convert embedded source sequence to have positional embeddings encoded within it

        return self.encoder(src, src_mask)

    def decode(self,
               encoder_output: torch.Tensor,
               src_mask: torch.Tensor,
               tgt: torch.Tensor,
               tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)  # embed the target sequences
        tgt = self.tgt_pos(tgt)  # add positional encoding to target embedded sequences
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # final projection layer with output vocab size
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
