import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = (
        32  # Num times to repeat the model i.e from RMSNorm to Feed Forward SwiGLU
    )
    n_heads: int = 32  # Num heads for queries
    n_kv_heads: Optional[int] = None  # Num heads for k and v
    vocab_size: int = -1  # will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5  # epsilon value in RMSNorm

    # Needed to KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def pre_compute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: 10000.0
):
    # According to Llama paper, the embedding dimension must be even
    assert head_dim % 2 == 0, "Head Dimension is not even"
    # Building the theta parameters
    # According to foruma that_i = 10000 ^ (-2(i-1)/dim) for i = [1,2,3..dim/2]
    # Shape: (Head_Dim/2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim/2)
    theta = theta ** -(theta_numerator / head_dim).to(device)

    # Construct Positions("m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len).to(device)

    # Multiply each theta by each position using user product
    # Shape: (Seq_Len) outerproduct* (Head_Dim/2) -> (Seq_Len, Head_Dim/2)
    freqs = torch.outer(m, theta).float()

    # We can compute complex numbers in polar form c = R * exp(i * m * theta), where R=1
    # so each element in the freq_complex will be like (a + ib) as complex numbers
    # Shape: (Seq_Len, Head_Dim/2) -> (Seq_Len, Head_Dim/2)
    freq_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freq_complex


def apply_rotatry_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):
    # H - Num Heads, B - Batch Size
    # We perform reshape, as the view_as_complex needs last dimension as 2
    # Shape: (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshaping freq_complex to match x_complex
    # Shape: (Seq_Len, Head_Dim/2) -> (1, Seq_Len, 1, Head_Dim/2)
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)

    # Multiplying x_complex with freq_complex i.e performing rotation
    # Shape: (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freq_complex

    # Viewing as Real instead Complex
    # Shape: (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)

    # reshaping
    # Shape: (B, Seq_Len, H, Head_DimHead_Dim/2, 2) - > (B, Seq_Len, H, Head_DimHead_Dim)
    x_out = x_rotated.reshape(*x.shape)

    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # epsilon so that division by zero doesnt occur
        self.eps = eps
        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # Mean is taken for squared elements for each sequence
        # rsqrt: 1/sqrt(x)
        # (B, Seq_Len, Dim) * (B, Seq_Len * 1) = (B, Seq_Len, Dim)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    else:
        return (
            # (B, Seq_Len, H_KV, 1, Head_Dim)
            x[:, :, :, None, :]  # creating a new dimension 1, before Head_Dim
            .expand(
                batch_size, seq_len, n_kv_heads, n_rep, head_dim
            )  # repeating the 1 Dim n_rep times
            .reshape(batch_size, seq_len, n_kv_heads, n_rep * head_dim)
            # reshaping to (B, Seq_Len, H_Q, Head_Dim) as H_KV * n_rep = H_Q
        )

# Performing a Grouped Multi Query Attention
class SelfAttention(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.n_heads_q = args.n_heads
        self.dim = args.dim
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = self.dim // self.n_heads_q

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_length, self.n_kv_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_length, self.n_kv_heads, self.head_dim)
        )

    def forward(self, x: torch.Tensor, start_pos: str, freq_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # Apply the Wq, Wk, Wv matrices to queries, keyes and values
        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk, xv = self.wk(x), self.wv(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q , Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_KV , Head_Dim)
        # The size of xk and xv can be less than the size of xq as H_Q * Head_Dim >= H_KV * Head_Dim
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Does not change the shape of the tensor
        xq = apply_rotatry_embeddings(xq, freq_complex)
        xk = apply_rotatry_embeddings(xk, freq_complex)

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xq
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # Retrieve all the cached keys and values so far
        # (B, Seq_len_KV, H_KV, Head_Dim)
        # the Seq_len_KV is equal to start_pos + 1, that is retrieve upto the start position
        keys = self.cache_k[:batch_size, 0 : start_pos + seq_len]
        values = self.cache_v[:batch_size, 0 : start_pos + seq_len]

        # Num of heads of KV may not match num of heads of Q
        # Hence repeat KV heads n_rep times to match the heads of Q
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transposing before calculating attention
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq,keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(),dim=-1).type_as(xq)
        
        # (B, H_Q, 1, Seq_Len_KV) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores,values)

        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = output.transpose(1,2).contiguous().view(batch_size,seq_len,-1)

        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)

class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x
    
class DecoderBlock(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.n_heads // self.dim

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Calculating RMSNorm BEFORE Attention
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        # Calculating RMSNorm AFTER Feedforward layer
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        h = x + self.attention(self.attention_norm(x), start_pos, freq_complex)
        out = self.feed_forward(self.ffn_norm(h))
        return out


class Transformer:
    def __init__(self, args: ModelArgs) -> None:
        super.__init__()

        assert self.args.vocab_size != -1, "Set the Vocabulary Size"
        self.args = args
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.args)

        self.layers = nn.ModuleList()

        for _ in range(self.n_layers):
            self.layers.append(DecoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=True)

        self.freqs_complex = pre_compute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # Shape: (B,Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only 1 token at a time can be processed"

        # Shape: (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve pairs of (m,theta) uptill the positions [start_pos,start_pos+seq_len]
        freq_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # Consecutively apply to all decoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freq_complex)

        h = self.norm(h)
        output = self.output(h)
        return output
