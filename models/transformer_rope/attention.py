"""
Originally forked from Andrej Karpathy's minGPT.

CS224N 2023-24: Homework 4

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """

    rope_cache = None
    i = torch.arange(dim // 2).unsqueeze(0)
    theta = 1 / (10000 ** (2 * (i) / dim))
    t = torch.arange(max_positions).unsqueeze(1)

    cosine_val = torch.cos(t * theta)
    sin_val = torch.sin(t * theta)

    rope_cache = torch.stack((cosine_val, sin_val), dim=-1)
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    rotated_x = None
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    new_x = x.view(*x.shape[:-1], half_dim, 2)
    if rope_cache.shape[-3] > new_x.shape[-3]:
        rope_cache = rope_cache[..., :new_x.shape[-3], :, :]

    complex_x = torch.view_as_complex(new_x)
    complex_rope = torch.view_as_complex(rope_cache)

    rotated_x = torch.view_as_real(complex_x * complex_rope)
    rotated_x = rotated_x.reshape(*x.shape)
    return rotated_x


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.rope = config.rope
        if self.rope:
            assert (config.n_embd % config.n_head) % 2 == 0
            rope_cache = None
            rope_cache = precompute_rotary_emb(
                config.n_embd // config.n_head, config.block_size)

            self.register_buffer("rope_cache", rope_cache)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C //
                             self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.rope:
            B, n_head, T, head_size = k.shape

            q = apply_rotary_emb(q.reshape(
                B * n_head, T, head_size), self.rope_cache).reshape(B, n_head, T, head_size)
            k = apply_rotary_emb(k.reshape(
                B * n_head, T, head_size), self.rope_cache).reshape(B, n_head, T, head_size)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalCrossAttention(nn.Module):
    """
    Modifications over the self-attention layer to handle two inputs and perform
    cross-attention between them.
    This follows the implementation of the self attention module with
    auto-regressive masking on (key).
    Manipulation of batch-size to allow for different batch size between the 
    two inputs, with broadcasting over to the higher batch size value.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x_kv, x_q):
        Bk, Tk, Ck = x_kv.size()
        Bq, Tq, Cq = x_q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        # keys of x1
        k = self.key(x_kv).view(Bk, Tk, self.n_head, Ck //
                                self.n_head).transpose(1, 2)  # (B, nh, Tk, hs)

        # query with x2
        q = self.query(x_q).view(Bq, Tq, self.n_head, Cq //
                                 # (B, nh, Tq, hs)
                                 self.n_head).transpose(1, 2)

        # values from x1
        v = self.value(x_kv).view(Bk, Tk, self.n_head, Ck //
                                  # (B, nh, Tk, hs)
                                  self.n_head).transpose(1, 2)

        # causal self-attention;  (B, nh, Tk, hs) x (B, nh, hs, Tq) -> (B, nh, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        B = max(Bk, Bq)

        att = att.masked_fill(self.mask[:, :, :Tq, :Tk] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, Tq, Tk) x (B, nh, Tk, hs) -> (B, nh, Tq, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, Tq, Cq)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
