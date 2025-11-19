from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

# This implementation is adapted from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)#
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 5, dim_head = 64, dropout = 0.):#
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)


## perceiver cross-transformer
def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

class Perceiver(nn.Module):
    def __init__(self, num_latents, latent_dim, input_dim, depth, heads, dim_head, latent_heads, latent_dim_head, attn_dropout=0., ff_dropout=0., weight_tie_layers=False, self_per_cross_attn=1):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.cross_heads = heads
        self.cross_dim_head = dim_head
        self.depth = depth
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))# here bsz?
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.latent_heads = latent_heads
        self.latent_dim_head = latent_dim_head
        get_cross_attn = lambda: PreNorm(self.latent_dim, Attention(self.latent_dim, self.input_dim, heads = self.cross_heads, dim_head = self.cross_dim_head, dropout = self.attn_dropout), context_dim = self.input_dim)
        get_cross_ff = lambda: PreNorm(self.latent_dim, FeedForward(self.latent_dim, dropout = self.ff_dropout))
        get_latent_attn = lambda: PreNorm(self.latent_dim, Attention(self.latent_dim, heads = self.latent_heads, dim_head = self.latent_dim_head, dropout = self.attn_dropout))
        get_latent_ff = lambda: PreNorm(self.latent_dim, FeedForward(self.latent_dim, dropout = self.ff_dropout))
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_embedds = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim)
        )  
    def forward(self, data, mask=None, return_embeddings=False):
        b = data.shape[0]
        len = data.shape[1]
        data = rearrange(data, 'b ... d -> b (...) d')
        x = repeat(self.latents, 'n d -> b n d', b = b)# b
        # layers
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x #
            x = cross_ff(x) + x
            
            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x
            
        if return_embeddings:
            return x

        return self.to_embedds(x)
        
   