import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor):
        # x -> (batch_size, num_tokens, d_in)

        q = self.W_q(x)  # (batch_size, num_tokens, d_out)
        k = self.W_k(x)
        v = self.W_v(x)

        attn_scores = q @ k.transpose(1, 2)
        print(
            "attn_scores:", attn_scores.shape
        )  # (num_tokens, num_tokens), since attn_scores are computed between all the different tokens with each other

        # dim=-1 because we want to normalize along the last dimension, for each token
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)

        print("attn_weights:", attn_weights.shape)

        return attn_weights @ v


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, context_length, qkv_bias=False) -> None:
        super().__init__()

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # used as the mask to set the upper right (excl. the main diagonal and set as -inf later)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        print("x", x.shape)
        batch, context_length, d_in = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)  # (batch, num_tokens, d_out)
        print("v", v.shape)

        attn_scores: torch.Tensor = q @ k.transpose(1, 2)
        print("attn_scores", attn_scores.shape)
        print("mask", self.mask.shape)

        # mask the "future" tokens since causal attention
        attn_scores.masked_fill_(self.mask.bool(), -torch.inf)

        # apply softmax to get logits into probs that sum to 1. multiply the square root of the d_out for numerical stability
        attn_weight = torch.softmax(
            attn_scores / q.shape[-1] ** 0.5, dim=-1
        )  # (batch, num_tokens, num_tokens)

        self.dropout(attn_weight)

        return attn_weight @ v  # (batch, num_tokens, d_out)


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, context_length, dropout, qkv_bias=False) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in, d_out, context_length=context_length, dropout=dropout, qkv_bias=qkv_bias
                )
                for _ in range(n_heads)
            ]
        )

    def forward(self, x):
        context_vecs = [head(x) for head in self.heads]
        return torch.stack(context_vecs, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(12)

    # attn_layer = SelfAttention(3, 3)
    attn_layer = CausalAttention(3, 4, context_length=6, dropout=0.5)

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ]
    )

    print("output", attn_layer(torch.stack([inputs, inputs])).shape)
