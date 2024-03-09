import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32 # Num times to repeat the model i.e from RMSNorm to Feed Forward SwiGLU
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
            self.layers.append(Encoder(args))

        self.norm = RMSNorm(args.dim, eps=args.eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=True)

        self.freqs_complex = pre_compute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )
