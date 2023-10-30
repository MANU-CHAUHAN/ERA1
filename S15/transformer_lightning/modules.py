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
        # d_model is model's dimension, raw sequences will be tranformed via this
        # here `d_model` is equal to all concatenated heads' dimension
        # (MODEL CAPACITY is defined by this)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

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

    def forward(self, q, k, v, mask):
        # use incoming q, k, v to get transformed q, k, v
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # breakdown or re-arange `d_model` into `h * d_k` parts via `view function` on all 3
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

        # send the re-aranged K, Q, V (for each HEAD now) and mask to calculate attention scores

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query=query,
                                                                     key=key,
                                                                     value=value,
                                                                     mask=mask,
                                                                     dropout=self.dropout)

        # combine all heads together now
        # reverse of previous transpose to restore original dimensions, also used `contiguous()`
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # final multiply with W_O
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)
