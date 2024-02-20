import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPT_conf:
    n_c: int = 16
    n_e: int = 32
    n_h: int = 4
    n_l: int = 4
    n_v: int = 27
    d: float = 0.1
    batch_size: int = 8


class LayerNorm(nn.Module):

    def __init__(self, n_e):
        super().__init__()
        self.e = 1e-8
        self.g = nn.Parameter(torch.ones(n_e))
        self.b = nn.Parameter(torch.zeros(n_e))
    
    def forward(self, X):
        mu = X.mean(dim=-1, keepdims=True)
        v = X.var(dim=-1, keepdims=True)
        X = (X - mu) / (torch.sqrt(v) + self.e)
        return self.g * X + self.b


class FeedForward(nn.Module):

    def __init__(self, n_e, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_e, 4 * n_e),
            nn.GELU(),
            nn.Linear(4 * n_e, n_e),
            nn.Dropout(d),
        )

    def forward(self, X):
        return self.net(X)


class Head(nn.Module):

    def __init__(self, n_e, sz):
        super().__init__()
        self.sz = sz
        self.q = nn.Linear(n_e, sz)
        self.k = nn.Linear(n_e, sz)
        self.v = nn.Linear(n_e, sz)
    
    def forward(self, X):
        Q = self.q(X)
        K = self.k(X)
        V = self.v(X)

        scaled_dot = Q @ torch.transpose(K, -1, -2) / self.sz ** 0.5
        AZ = scaled_dot.masked_fill(torch.tril(scaled_dot) == 0, float('-inf'))
        return F.softmax(AZ, dim=-1) @ V


class MHA(nn.Module):

    def __init__(self, n_e, n_h):
        super().__init__()
        sz = n_e // n_h
        self.heads = nn.ModuleList([Head(n_e, sz) for _ in range(n_h)])
        self.W_o = nn.Linear(sz * n_h, n_e)
    
    def forward(self, X):
        o = torch.cat([h(X) for h in self.heads], dim=-1)
        return self.W_o(o)


class Block(nn.Module):

    def __init__(self, n_e, n_h, d):
        super().__init__()
        self.norm_z = LayerNorm(n_e)
        self.mha = MHA(n_e, n_h)
        self.norm_a = LayerNorm(n_e)
        self.mlp = FeedForward(n_e, d)
    
    def forward(self, x):
        x = x + self.mha(self.norm_z(x))
        x = x + self.mlp(self.norm_a(x))
        return x


class GPTModel(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.E = nn.Embedding(self.conf.n_v, self.conf.n_e)
        self.P = nn.Embedding(self.conf.n_c, self.conf.n_e)
        self.layers = nn.Sequential(*[Block(self.conf.n_e, self.conf.n_h, self.conf.d) for _ in range(self.conf.n_l)])
        self.normalize_o = LayerNorm(self.conf.n_e)
        self.mlp = nn.Sequential(
            nn.Dropout(self.conf.d),
            nn.Linear(self.conf.n_e, self.conf.n_v),
        )
        self.apply(self._w_and_b)
        self.model_state = {'best_loss': float('inf')}

    def _w_and_b(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, y=None):
        T = len(x)
        e = self.E(x) + self.P(torch.arange(T))

        a = self.layers(e)
        Z =  self.mlp(self.normalize_o(a))
        B, T, C = Z.shape

        if y is not None:
            y = y.view(B*T)
            Z = Z.view(B*T, C)
            loss = F.cross_entropy(Z, y)
        else:
            loss = None
        return Z, loss

    def create(self, p, max_new_tokens):
        """ p -- a prompt-like """
        for _ in range(max_new_tokens):
            p = p[:, - self.conf.n_c:]
            Z, loss = self(p)
            Z = Z[:, -1, :]
            probs = F.softmax(Z, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            
            p = torch.cat((p, new_token), dim=1)
        return p