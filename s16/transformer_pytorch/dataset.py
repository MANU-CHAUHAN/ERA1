import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int8)
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int8)
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int8)

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
        # enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # consider SOS and EOS
        #
        # # consider SOS only as EOS is in the label
        # dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # make sure the padded sequence is not negative, seq_len should be calculated separately beforehand for max value
        # if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
        #     raise ValueError("Sentence too long")

        # Add <s> and </s> token and <pad>, with calculation for number of <pad>s needed
        # Encoder structure is: <SOS>, <input_token_ids>, <EOS>, <PAD>, ... <PAD>
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int32),
                self.eos_token,
                # torch.tensor([self.pad_token] *
                #              enc_num_padding_tokens, dtype=torch.int32)
            ], dim=0
        )

        # Decoder: add only <SOS>, <input_token_ids> and <PAD>...<PAD> , as <EOS> is to be predicted by the decoder
        # <SOS>, <input_token_ids>, <PAD>, ... <PAD>
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int32),
                # torch.tensor([self.pad_token] *
                #              dec_num_padding_tokens, dtype=torch.int32)
            ], dim=0
        )

        # for label, add only <EOS> token with input and pads
        # <decoder input token ids>, <EOS>, <PAD>, ... <PAD>
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                # torch.tensor([self.pad_token] *
                #              dec_num_padding_tokens, dtype=torch.int32)

            ]
        )

        # checks for dimensions
        # assert encoder_input.size(0) == self.seq_len
        # assert decoder_input.size(0) == self.seq_len
        # assert label.size(0) == self.seq_len

        # enc_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        # dec_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

        r = {
            "encoder_input": encoder_input,  # seq_len
            "decoder_input": decoder_input,  # seq_len
            # (1, 1, seq_len)
            # "encoder_mask" : enc_mask,
            # "decoder_mask" : dec_mask,
            # (1, seq_len) & (1, seq_len, seq_len)
            "label"        : label,  # (seq_len)
            "src_text"     : src_txt,
            "tgt_text"     : tgt_txt,
            "encoder_token_len": len(encoder_input),
            "decoder_token_len": len(decoder_input),
            'pad_token': self.pad_token
        }

        return r


def causal_mask(size):
    # mask with upper triangle, with an extra size=1 removed above the diagonal, specified by diagonal=1
    """
    This is commonly used in sequence-to-sequence models like transformers to ensure that the decoder only attends to positions
      that have been previously decoded, preventing it from looking into the future positions.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


"""
Notes:
Masking explained (https://stackoverflow.com/a/59713254/3903762):

The short answer is - we need masking to make the training parallel. And the parallelization is good as it allows the model to train faster.

Here's an example explaining the idea. Let's say we train to translate "I love you" to German. 
  The encoder works in parallel mode - it can produce vector representation of the input sequence ("I love you") within a constant number of steps
    (i.e. the number of steps doesn't depend on the length of the input sequence).

Let's say the encoder produces the numbers 11, 12, 13 as the vector representations of the input sequence. In reality these vectors will be much longer but for simplicity we use the short ones. Also for simplicity we ignore the service tokens, like - beginning of the sequence, - end of the sequence and others.

During the training we know that the translation should be "Ich liebe dich" (we always know the expected output during the training). Let's say the expected vector representations of the "Ich liebe dich" words are 21, 22, 23.

If we make the decoder training in sequential mode, it'll look like the training of the Recurrent Neural Network. The following sequential steps will be performed:

Sequential operation #1. Input: 11, 12, 13.
Trying to predict 21.
The predicted output won't be exactly 21, let's say it'll be 21.1.
Sequential operation #2. Input: 11, 12, 13, and also 21.1 as the previous output.
Trying to predict 22.
The predicted output won't be exactly 22, let's say it'll be 22.3.
Sequential operation #3. Input 11, 12, 13, and also 22.3 as the previous output.
Trying to predict 23.
The predicted output won't be exactly 23, let's say it'll be 23.5.
This means we'll need to make 3 sequential operations (in general case - a sequential operation per each input). Also we'll have an accumulating error on each next iteration. Also we don't use attention as we only look to a single previous output.

As we actually know the expected outputs we can adjust the process and make it parallel. There's no need to wait for the previous step output.

Parallel operation #A. Inputs: 11, 12, 13.
Trying to predict 21.
Parallel operation #B. Inputs: 11, 12, 13, and also 21.
Trying to predict 22.
Parallel operation #C. Inputs: 11, 12, 13, and also 21, 22.
Trying to predict 23.
This algorithm can be executed in parallel and also it doesn't accumulate the error. And this algorithm uses attention (i.e. looks to all previous inputs) thus has more information about the context to consider while making the prediction.

And here is where we need the masking. The training algorithm knows the entire expected output (21, 22, 23). It hides (masks) a part of this known output sequence for each of the parallel operations.

When it executes #A - it hides (masks) the entire output.
When it executes #B - it hides 2nd and 3rd outputs.
When it executes #C - it hides 3rd output.
Masking itself is implemented as the following (from the original paper):

We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections

Note: during the inference (not training) the decoder works in the sequential (not parallel) mode as it doesn't know the output sequence initially. But it's different from RNN approach as Transformer inference still uses self-attention and looks at all previous outputs (but not only the very previous one).

-------
=======

In Transformers, particularly in the decoder, masking is used to ensure that during self-attention,
 each position can only attend to positions that have been previously decoded (to the left of the current position)
 to maintain the causal nature of the model. This is known as the "causal mask."

Suppose decoder_input is a tensor representing a sequence of tokens with a length of 5, where the tokens are [2, 4, 3, 0, 1], and self.pad_token is 0. 

So, decoder_input looks like this:
 decoder_input = torch.tensor([2, 4, 3, 0, 1])

 1. (decoder_input != self.pad_token): This part of the code creates a Boolean tensor by comparing each element of decoder_input with self.pad_token. 
 Since self.pad_token is 0, the comparison is as follows:
 [2, 4, 3, 0, 1] != 0

 Result is: [True, True, True, False, True]

    This tensor indicates which positions in decoder_input are not equal to 0.
 
 2. `unsqueeze(0)` : [[True, True, True, False, True]]

 3. `.int()`: [[1, 1, 1, 0, 1]]

 4. `causal_mask(decoder_input.size(0))`: This part calls the causal_mask function with the size of decoder_input, which is 5.
    
    The causal_mask function generates a mask with an upper triangular shape, where elements above the main diagonal are 1,
      and elements below the main diagonal are 0, with an extra size=1 removed above the diagonal (diagonal=1).

    For size=5, the mask generated by causal_mask would look like this:
    [
    [0, 1, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]
    ]

5. `&`: a bitwise AND operation between the integer tensor created in step 3 and the causal mask created in step 4:

This mask is applied to the self-attention mechanism to ensure that a position can attend only to positions that have already been processed. 
When computing attention scores, the mask is added to the raw scores as follows:

For positions that have 0 in the mask (below the main diagonal), the corresponding attention scores are set to negative infinity (or a large negative value).

For positions that have 1 in the mask (above the main diagonal), the attention scores remain unchanged.

The softmax function is then applied to these modified attention scores, making the masked-out positions effectively contribute zero to the final attention distribution.

 
"""
