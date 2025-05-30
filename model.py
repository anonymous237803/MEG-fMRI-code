import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import zscore_tensor, hilbert_torch

# -----------------------------------------------------------------------------
# Utility: generate a causal mask of size (T, T)
# -----------------------------------------------------------------------------
def causal_mask(T: int, device=None) -> torch.Tensor:
    # mask[i,j] = 0 if j<=i else -inf
    m = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    m = m.masked_fill(m == 1, float("-inf")).masked_fill(m == 0, float(0.0))
    return m  # to be passed as src_mask


# -----------------------------------------------------------------------------
# The local transformer encoder layer
# -----------------------------------------------------------------------------
class LocalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, window, chunk_size=256, dropout=0.1):
        """
        d_model: hidden size
        nhead:   number of heads
        window:  how many past tokens each position can attend to
        """
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.window = window
        self.chunk = chunk_size

        # project once to Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Shaw-style relative positional bias: one bias per relative distance (0..window-1) and per head
        self.rel_pos_bias = nn.Parameter(torch.zeros(window, nhead))
        nn.init.zeros_(self.rel_pos_bias)

    def forward(self, x):
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)
        """
        B, T, d_model = x.size()
        # 1) project to Q,K,V and reshape
        qkv = self.qkv_proj(x)  # (B, T, 3·d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, d_model)
        # reshape for heads
        q = q.view(B, T, self.nhead, self.head_dim)
        k = k.view(B, T, self.nhead, self.head_dim)
        v = v.view(B, T, self.nhead, self.head_dim)

        # 2) pad K and V on the time dimension so early positions still have window entries
        #    pad on the left with (window-1) zeros
        pad = self.window - 1
        zeros = k.new_zeros((B, pad, self.nhead, self.head_dim))
        k_pad = torch.cat([zeros, k], dim=1)        # (B, T+pad, nhead, head_dim)
        v_pad = torch.cat([zeros, v], dim=1)

        # prepare container for the attended context
        context = k.new_empty((B, T, self.nhead, self.head_dim))
        mask_local = torch.arange(self.window, device=x.device).view(1, self.window) < (pad - torch.arange(T, device=x.device)).view(T, 1)  # (T, W) bool
        
        # process in chunks of length L = chunk_size
        for start in range(0, T, self.chunk):
            end = min(T, start + self.chunk)

            # select query block
            q_block = q[:, start:end]                   # (B, L, H, D)

            # select the necessary K/V slice of length L + pad
            k_block = k_pad[:, start : end + pad]       # (B, L+pad, H, D)
            v_block = v_pad[:, start : end + pad]

            # unfold + permute into (B, C, W, H, D)
            k_windows = k_block.unfold(1, self.window, 1).permute(0, 1, 4, 2, 3)
            v_windows = v_block.unfold(1, self.window, 1).permute(0, 1, 4, 2, 3)

            # compute scores & attention
            # q_block.unsqueeze(2): (B, C, 1, H, D)
            # kw: (B, C, W, H, D)
            scores = (q_block.unsqueeze(2) * k_windows).sum(-1)
            scores = scores / math.sqrt(self.head_dim)
            
            # rel_pos_bias: (window, nhead) -> (1, 1, W, H)
            scores = scores + self.rel_pos_bias.unsqueeze(0).unsqueeze(0)
            
            # **mask out left-pad** before softmax
            mask_block = mask_local[start:end]                    # (L, W), True means it is padded
            mask_block = mask_block.view(1, end-start, self.window, 1)      # (1, L, W, 1)
            scores = scores.masked_fill(mask_block, float('-inf'))
    
            attn = F.softmax(scores, dim=2)
            attn = self.dropout(attn)

            # weighted sum
            # attn.unsqueeze(-1): (B, C, W, H, 1)
            # vw:                 (B, C, W, H, D)
            context_block = (attn.unsqueeze(-1) * v_windows).sum(2)  # (B, C, H, D)

            # write back
            context[:, start:end] = context_block

        # merge heads & final linear
        context = context.contiguous().view(B, T, d_model)
        return self.out_proj(context)


class LocalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, window, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = LocalSelfAttention(d_model=d_model, nhead=nhead, window=window, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        attn_out = self.self_attn(x)
        x = x + attn_out
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        return x


# -----------------------------------------------------------------------------
# The full model
# -----------------------------------------------------------------------------
class TransformerSourceModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        window: int = 500,
        lead_field: torch.Tensor = None,
        ds_freq: int = 50,
        tr: int = 2,
        L: int = 10,
        source_power: bool = True,
        fmri_noise: bool = True,
    ):
        super().__init__()
        n_neurons = lead_field.shape[1]
        self.ds_factor = ds_freq * tr
        self.source_power = source_power
        self.fmri_noise = fmri_noise
        self.register_buffer("lead_field", lead_field)  # (n_channels, n_neurons)

        # per‐neuron HRF conv
        self.L = L
        self.hrf_conv = nn.Conv1d(in_channels=n_neurons, out_channels=n_neurons, kernel_size=self.L, groups=n_neurons, bias=False)

        # assemble layers
        self.input_proj = nn.Linear(embed_dim, d_model)
        encoder_layer = [LocalTransformerEncoderLayer(d_model=d_model, nhead=nhead, window=window, dim_ff=dim_ff, dropout=dropout) for _ in range(num_layers)]
        self.transformer = nn.Sequential(*encoder_layer)
        self.to_neurons = nn.Linear(d_model, n_neurons)

    def forward(self, x_embed: torch.Tensor, return_nonz=True) -> dict:
        """
        x_embed: (B, T, embed_dim)
        returns:
         - neurons: (B, T, n_neurons)
         - meg_pred: (B, T, n_channels)  # after lead field
         - fmri_pred: (B, T/100-5, n_neurons)  # after power, pool, HRF conv
        """

        # embed
        x = self.input_proj(x_embed)

        # transformer (causal)
        x = self.transformer(x)

        # neuron time series
        neurons = self.to_neurons(x)  # (B, T, N)

        # MEG pred
        meg_pred = torch.einsum("btn,sn->bts", neurons, self.lead_field)  # s=n_channels
        neurons_return = neurons.clone()
        
        # zscore neurons
        neurons = zscore_tensor(neurons, dim=1)
        
        # Gaussian noise regularisation before fMRI
        if self.training and self.fmri_noise:    
        # if False:
            neurons_noisy = neurons + 0.1 * torch.randn_like(neurons)
            neurons_noisy = F.dropout(neurons_noisy, p=0.5)
        else:
            neurons_noisy = neurons
        
        # fMRI conv
        nm = neurons_noisy.transpose(1, 2)  # (B, N, T)
        if self.source_power:  # get power
            nm = hilbert_torch(nm.cpu(), dim=2)
            nm = nm.abs()
            nm.pow_(2)
            nm = nm.to(x.device)
        nm = F.avg_pool1d(nm, kernel_size=self.ds_factor, stride=self.ds_factor)  # (B, N, T/100)
        nm = F.pad(nm, (self.L - 1 - 5, 0))  # (B, N, T/100+L-1-5)
        fmri_pred = self.hrf_conv(nm)  # (B, N, T/100-5)
        fmri_pred = fmri_pred.transpose(1, 2)  # (B, T/100-5, N)

        if return_nonz:
            return neurons_return, meg_pred, fmri_pred
        else:  
            return neurons, meg_pred, fmri_pred
