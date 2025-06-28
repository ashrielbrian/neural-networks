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
        )  # (batch_size, num_tokens, num_tokens), since attn_scores are computed between all the different tokens with each other

        # dim=-1 because we want to normalize along the last dimension, for each token. i.e., for each token, how important are all
        # these other tokens, summed to 1.
        attn_weights = torch.softmax(
            attn_scores / k.shape[-1] ** 0.5, dim=-1
        )  # (batch_size, num_tokens, num_tokens)

        print("attn_weights:", attn_weights.shape)
        return attn_weights @ v  # (batch_size, num_tokens, d_out)


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


class EfficientMultiHeadCausalAttention(nn.Module):
    """Uses batched matrix multiplication instead of sequentially multiplying each attention head separately."""

    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.5, qkv_bias=False) -> None:
        # d_in - token embedding size
        # d_out = num_heads * head_dim, i.e. the total output dimension of the multihead. in the literature,
        # the each attention head  takes an equal portion of d_out, and after causal self-attn of each head,
        # the outputs of each head is concatenated together to give d_out again
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # optional projection layer
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        queries: torch.Tensor = self.W_query(x)
        keys: torch.Tensor = self.W_key(x)
        values: torch.Tensor = self.W_value(x)  # (batch_size, num_tokens, d_in)

        # here, we split into each head by unrolling d_in -> num_heads * head_dim
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # transpose to (batch_size, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(
            2, 3
        )  # (batch_size, num_heads, num_tokens, num_tokens)

        # masking truncated to num_tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # apply softmax on the masked scores to get the attn weights
        attn_weights = torch.softmax(attn_scores / (self.head_dim**0.5), dim=-1)
        self.dropout(attn_weights)  # (batch_size, num_heads, num_tokens, num_tokens)

        context_vec = attn_weights @ values  # (batch_size, num_heads, num_tokens, head_dim)
        context_vec = context_vec.transpose(1, 2)  # (batch_size, num_tokens, num_heads, head_dim)

        # essentially concatenating the outputs from all the attention heads
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)

        return self.out_proj(context_vec)


def test_causal_attn(inputs):
    torch.manual_seed(12)
    attn_layer = CausalAttention(3, 4, context_length=6, dropout=0.5)

    print("output", attn_layer(torch.stack([inputs, inputs])).shape)


def test_emha(inputs):
    torch.manual_seed(42)

    batch = torch.stack([inputs, inputs], dim=0)
    batch_size, num_tokens, d_in = batch.shape

    print(batch_size, num_tokens, d_in)

    d_out, num_heads = 4, 2

    emha = EfficientMultiHeadCausalAttention(
        d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=num_tokens
    )

    out = emha(batch)

    print(out)
    print("emha shape:", out.shape)


if __name__ == "__main__":
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

    # test_causal_attn(inputs)
    test_emha(inputs)
