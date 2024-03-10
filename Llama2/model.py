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
            self.layers.append(Decoder(args))

        self.norm = RMSNorm(args.dim, eps=args.eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=True)

        self.freqs_complex = pre_compute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )
