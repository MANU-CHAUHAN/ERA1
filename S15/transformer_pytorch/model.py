import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # learnable param
        self.bias = nn.Parameter(torch.zeros(1))  # again, a learnable param

    def forward(self, x):
        # x is of shape: (batch, seq_len, hidden_size)
        # keep dims = True for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # becomes (batch, seq_len, 1)

        std = x.std(dim=-1, keepdim=True)  # becomes (batch, seq_len, 1)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # idea is to get d_model dimension and expand these to higher number, usually twice or more,
        # then shrink back to original d_model dimension
        self.linear1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # concat of heads
        # x is (batch, seq_len, d_model) -> to (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        # here d_ff is usually twice or more of d_model
        # (d_model is concatenated heads say 512 of 8 heads of 64 dim each)

        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        # d_model is model's dimension, raw sequences will be transformed via this
        # here `d_model` is equal to all concatenated heads' dimension 
        # (MODEL CAPACITY is defined by d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # input vocab size

    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        # here we multiply with sqrt(d_model) to scale the embeddings (as per the paper)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 dropout: float) -> None:
        super().__init__()

        # The positional encodings that are used to add position information to the input embeddings

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # a vector of shape (seq_len) -> (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000) / d_model))

        # apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))

        # apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))

        # add batch dimension to positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # register positional encoding as buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (batch, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # note: requires_grad_ vs requires_grad
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 h: int,
                 dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "`d_model` must be divisible by `h`"

        self.d_k = d_model // h  # dim of each head

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query weights
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Output matrix weights

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # get the incoming head's (last) dimension, let's call it mini-d_model
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        #   # eg Q = 6x512  K = 6x512  (seq_len, embd_dim)
        # scale as per the paper
        attention_scores = (query @ key.transpose(-2, -1)) / (math.sqrt(d_k))

        # if mask is provided, set those to very low value, representing -inf, to avoid looking ahead
        if mask is not None:
            _MASKING_VALUE = -1e+9 if attention_scores.dtype == torch.float32 else -1e+4
            attention_scores.masked_fill_(mask == 0, _MASKING_VALUE)

        # apply softmax on the attention scores
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # multiply attention scores with Value
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return is tuple of final weighted value and attention scores (incase needed later)
        return attention_scores @ value, attention_scores
        # # Efficient implementation equivalent to the following:
        # scale_factor = 1 / math.sqrt(d_k)
        # attn_mask = mask.masked_fill(not mask, -float('inf')) if mask.dtype == torch.bool else mask
        # attn_weight = torch.softmax((query @ key.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1)
        # attn_weight = dropout(attn_weight)
        # return attn_weight @ value, attn_weight

    def forward(self, q, k, v, mask):
        # use incoming q, k, v to get transformed q, k, v
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # breakdown or re-arrange `d_model` into `h * d_k` parts via `view function` on all 3
        #
        # observe the transpose here of dims 1 and 2 which is seq_len, h --> h, seq_len 
        #   # basically re-arranging the dimensions to have batch and then head number
        #   # where `seq_len, d_k` represents the concerned matrix FOR each head now, given any `h`
        #   # this is where d_model % h == 0 comes into play
        #   
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)

        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # send the re-arranged K, Q, V (for each HEAD now) and mask to calculate attention scores

        # x, self.attention_scores = MultiHeadAttentionBlock.attention(query=query,
        #                                                              key=key,
        #                                                              value=value,
        #                                                              mask=mask,
        #                                                              dropout=self.dropout)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            x = F.scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=mask, dropout_p=0.1)

        # combine all heads together now
        # reverse of previous transpose to restore original dimensions, also used `contiguous()`
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # final multiply with W_O
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)


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
        # here q, k, v all are same (x) for Encoder
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        # next we send this processed block to feed forward, which has residual connection too
        # 
        # original x has been updated already, now x refers to incoming data from previous block of self-attention
        # block in Encoder

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

        # this is the cross attention block, getting Q from itself while K and V comes from encoded output from Encoder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))

        # Feed Forward block as the third block
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
    src_pos = PositionalEncoding(d_model=d_model, seq_len=src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(d_model=d_model, seq_len=tgt_seq_len, dropout=dropout)

    # create list to hold Encoder blocks
    encoder_blocks = []
    for _ in range(N):  # repeat for N
        # create or initialize sub-layers for current Encoder block
        encoder_self_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # create 1 instance of Encoder block
        encoder_block = EncoderBlock(self_attention_block=encoder_self_attention,
                                     feed_forward_block=feed_forward_block,
                                     dropout=dropout)

        # add this 1 block to Encoder block list
        encoder_blocks.append(encoder_block)

    # create a list to hold Decoder blocks
    decoder_blocks = []
    for _ in range(N):  # have `N` total blocks
        # initialize sub-layers for each block
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)

        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)

        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)

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
    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size)

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
