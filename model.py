import math
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    input_size: int = 128
    causal: bool = True
    n_layer: int = 3
    n_head: int = 1
    n_embd: int = 32
    dropout: float = 0
    softmax: str = "Quiet" # "Quiet" or "Loud"
    positional: bool = False
    vocab_size: int = 75000
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster



def QuietSoftmax(x, dim=None):
    return torch.exp(x) / (1 + torch.exp(x).sum(dim=dim, keepdim=True))


class LayerNorm(nn.Module):
    """Layer norm with optional bias"""
    def __init__(self, n_dim, bias=True):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None
        
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class SelfAttention(nn.Module):

    def __init__(self, config):
        super(SelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.softmax = QuietSoftmax if config.softmax == "Quiet" else F.softmax


        self.causal = config.causal
        
    
    def forward(self, x):

        
        B, T, C = x.shape # B Batch, T num agents, C is hidden size

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Transpose everything into shape (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)


        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))


        return y  #, out_attn
    


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        in_x = x 
        # x, attn = self.attn(self.ln_1(x))
        x = self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x + in_x))

        return x#, attn
    




class GPT(nn.Module):

    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            embedder = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.embedder.weight = self.lm_head.weight
        
        self.positional = config.positional
        self.input_size = config.input_size

        if self.positional:
            self.pos = nn.Parameter(torch.zeros(1, self.input_size, config.n_embd))
            self.pos_dropout = nn.Dropout(config.dropout)



    def forward(self, raw_input, targets=None):
        
        x = self.transformer.drop(self.transformer.embedder(raw_input))

        if self.positional:
            x = x + self.pos
            x = self.pos_dropout(x)

        # layerwise_attn = []

        for block in self.transformer.h:
            # x, attn = block(x)
            x = block(x)
            # layerwise_attn.append(attn)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x)

        return logits, loss
    

    def generate(self, raw_input, max_new_steps=100):

        for _ in range(max_new_steps):
            coords, _, _ = self(raw_input)
            
            raw_input = torch.cat((raw_input, coords), dim=2)

        return raw_input


def train_model(model, train_pairs, epochs):

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_history = []

    for epoch in range(epochs):

        running_loss = 0

        for train_pair in train_pairs:
            
            optim.zero_grad()
            x, y = train_pair
            coords, loss = model(x.unsqueeze(0), targets=y)
            loss.backward()
            optim.step()
            running_loss += loss.item()

        epoch_loss = running_loss/len(train_pairs)
        print(f"{epoch=} : {epoch_loss:.3f}")
        loss_history.append(epoch_loss)

    
    return loss_history

        



            
if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)
    print(model)
