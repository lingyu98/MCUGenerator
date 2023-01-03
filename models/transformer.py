import math
import torch
import torch.nn as nn
from .utils import get_clones, NewGELU

#TODO: Adapt attention module to accept custom padding mask/causal mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class FixedPositionalEncoder(nn.Module):
    # FIxed postional encoder
    def __init__(self, d_model, max_T=80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_T, d_model)
        for pos in range(max_T):
            for i in range(0, model, 2):
                pe[pos, i] = math.sin(pos / (1e4 ** (2*i / d_model)))
                pe[pos, i+1] = math.cos(pos / (1e4 ** (2*(i+1) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        x = torch.tensor(self.pe[:, :seq_len], requires_grad=False).to(device)
        return x

class LearnedPositionalEncoder(nn.Module):
    # learned postional encoder
    def __init__(self, d_model, max_T=80):
        super().__init__()
        self.embed = nn.Embedding(max_T, d_model)
        self.max_T = max_T

    def forward(self, x):
        # x [B, t, E]
        pos = torch.arange(0, self.max_T, dtype=torch.long).to(device).unsqueeze(0)
        pos = self.embed(pos)
        return pos

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, max_T, dropout=0.1):
        super().__init__()
        torch._assert(d_model%heads==0, 'd_model=%d not dividable by nheads=%d'%(d_model, heads))
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.max_T = max_T

        self.K, self.Q, self.V, self.out = [nn.Linear(d_model, d_model)] * 4
        self.dropout = nn.Dropout(dropout)

    def attention(self, k, q, v, mask, causal):
        # k [B, H, T, E]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])

        if causal:
            causal_mask = torch.ones(self.max_T, self.max_T)
            causal_mask = torch.tril(causal_mask).view(1, 1, self.max_T, self.max_T) # [1, 1, T, T]
            scores = scores.masked_fill(causal_mask==0, float('-inf'))

        if mask is not None:
            torch._assert(mask.shape[-2:]==scores.shape[-2:], 'mask shape must be [_, _, T, T]')
            scores = scores.masked_fill(mask==0, float('-inf')) # [B, TxT]

        scores = scores.softmax(-1)
        scores = self.dropout(scores)
        out = scores @ v
        return out

    def forward(self, k, q, v, mask=None, causal=False):

        torch._assert(k.shape[1]==self.max_T, 'input seq_len must be padded to %d' %self.max_T)

        B = k.shape[0]

        k = self.K(k).view(B, self.max_T, self.h, self.d_k).transpose(1, 2) # B, H, T, Dk
        q = self.Q(q).view(B, self.max_T, self.h, self.d_k).transpose(1, 2)
        v = self.V(v).view(B, self.max_T, self.h, self.d_k).transpose(1, 2)

        if mask is not None:
            torch._assert(mask.shape == torch.Size([B, self.max_T]), 'mask shape must be [B, T]')
            mask = mask.float()
            mask =  mask.unsqueeze(1).transpose(-2, -1) @ mask.unsqueeze(1) # [B, T, T]
            mask = mask.unsqueeze(1) #[B, 1, T, T]

        scores = self.attention(k, q, v, mask, causal)
        concat = scores.transpose(1, 2).contiguous().view(B, self.max_T, self.d_model)
        out = self.dropout(self.out(concat))
        return out

class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            NewGELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.layers(x)

class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, max_T=80):
        super().__init__()
        self.norm1, self.norm2 = [nn.LayerNorm(d_model)] * 2
        self.mha = MultiHeadAttention(heads, d_model, max_T)
        self.mlp = MLP(d_model)

    def forward(self, x, mask=None):
        x = x + self.mha(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, heads, d_model, max_T=80):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(d_model)] * 3
        self.mha1, self.mha2 = [MultiHeadAttention(heads, d_model, max_T)] * 2
        self.mlp = MLP(d_model)

    def forward(self, x, e_out, src_mask=None, tgt_mask=None):
        x = x + self.mha1(self.norm1(x), self.norm1(x), self.norm1(x), mask=tgt_mask, causal=True)
        x = x + self.mha2(k=self.norm2(e_out), q=self.norm2(x), v=self.norm2(e_out), mask=src_mask)
        x = x + self.mlp(self.norm3(x))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, max_T=80):
        super().__init__()
        self.N = N
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = LearnedPositionalEncoder(d_model, max_T)
        self.dropout = nn.Dropout(0.1)

        self.layers = get_clones(EncoderLayer(heads, d_model, max_T), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        tok_emb = self.tok_emb(src)
        pos_emb = self.pos_emb(src)
        x = self.dropout(tok_emb + pos_emb)

        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, max_T=80):
        super().__init__()
        self.N = N
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = LearnedPositionalEncoder(d_model, max_T)
        self.dropout = nn.Dropout(0.1)

        self.layers = get_clones(DecoderLayer(heads, d_model, max_T), N)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, tgt, e_out, src_mask=None, tgt_mask=None):
        tok_emb = self.tok_emb(tgt)
        pos_emb = self.pos_emb(tgt)
        x = self.dropout(tok_emb + pos_emb)
        
        for i in range(self.N):
            x = self.layers[i](x, e_out, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, N, heads, max_T=80):
        super().__init__()
        self.enc = Encoder(src_vocab, d_model, N, heads, max_T)
        self.dec = Decoder(tgt_vocab, d_model, N, heads, max_T)
        self.out = nn.Linear(d_model, tgt_vocab)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        e_out = self.enc(src, src_mask)
        d_out = self.dec(tgt, e_out, src_mask, tgt_mask)
        out = self.out(d_out)
        return out

class GPT_Block(nn.Module):
    def __init__(self, heads, d_model, max_T=80):
        super().__init__()
        self.norm1, self.norm2 = [nn.LayerNorm(d_model)] * 2
        self.mha = MultiHeadAttention(heads, d_model, max_T)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.mha(self.norm1(x), self.norm1(x), self.norm1(x), mask=None, causal=True)
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, max_T):
        super().__init__()
        self.N = N
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = LearnedPositionalEncoder(d_model, max_T)
        self.dropout = nn.Dropout(0.1)

        self.layers = get_clones(GPT_Block(heads, d_model, max_T), N)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, src):
        tok_emb = self.tok_emb(src)
        pos_emb = self.pos_emb(src)
        x = self.dropout(tok_emb + pos_emb)
        for i in range(self.N):
            x = self.layers[i](x)
        x = self.norm(x)
        logits = self.out(x)
        return logits

    def predict_next(self, src):
        out = self.forward(src)[:, 0, :]
        return out

if __name__ == '__main__':
    #model = Transformer(100, 100, 32, 1, 2, 10)
    model = GPT(100, 32, 1, 2, 10)
    x = torch.tensor([[1,2,3,4,5,11,11,11,11,11],[6,7,8,9,11,11,11,11,11,11]])
    src_mask = x!=0
    out = model(x)
    print(out.shape)
