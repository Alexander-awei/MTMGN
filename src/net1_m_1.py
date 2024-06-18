# Several functions borrowed from https://github.com/ermongroup/CSDI/blob/main/diff_models.py
# And also from Martinez et al: https://github.com/una-dinosauria/human-motion-prediction
from functools import partial
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# make time t into 128-dim embedding vector.
# borrowed from https://github.com/ermongroup/CSDI/blob/main/diff_models.py 
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        # Codes from CSDI
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
# 用于将扩散部步数（diffusion-step）映射到低维嵌入空间

# Positional Encoding modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
# 为输入序列的每个位置添加一个固定的位置编码，通过随即丢弃一些位置编码，以减少过拟合




class Series_Denoiser(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len
        self.diff_steps = diff_steps

        self.spatial_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                        dim_feedforward=qkv_dim, activation='gelu')
        self.spatial_TF = nn.TransformerEncoder(self.spatial_layer, num_layers=num_layers)
        self.spatial_inp_fc = nn.Linear(prefix_len + pred_len, qkv_dim)
        self.spatial_out_fc = nn.Linear(qkv_dim, prefix_len + pred_len)

        self.temporal_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                         dim_feedforward=qkv_dim, activation='gelu')
        self.temporal_TF = nn.TransformerEncoder(self.temporal_layer, num_layers=num_layers)
        self.temporal_inp_fc = nn.Linear(input_dim, qkv_dim)
        self.temporal_out_fc = nn.Linear(qkv_dim, input_dim)

        self.pos_encoder = PositionalEncoding(qkv_dim)

        self.step_temporal_encoder = DiffusionEmbedding(diff_steps, qkv_dim)
        self.step_spatial_encoder = DiffusionEmbedding(diff_steps, qkv_dim)

        # add condition
        dim = 32
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, 2)
        )

        block_klass = partial(ResnetBlock, groups=2)
        block_klass_cond = partial(block_klass, time_emb_dim=4)

        self.block1 = block_klass_cond(2, 46)
        # self.block2 = block_klass_cond(4, 8)


    def forward(self, noise, x, t, cond):
        t_cond = self.time_mlp(t)
        t_cond_mix = torch.cat((t_cond, cond), dim=-1)

        step_spatial_embed = self.step_spatial_encoder(t)
        step_temporal_embed = self.step_temporal_encoder(t)
        window = torch.cat([x, noise], dim=0)

        window = window.permute(1, 0, 2)
        window = self.block1(window, t_cond_mix)
        window = window.squeeze(-1).squeeze(-1).permute(1, 0, 2)
        # x = self.block2(x, t_cond_mix)

        spatial = self.spatial_inp_fc(window.permute(2, 1, 0)) + step_spatial_embed  # L B D -> D B L
        spatial = self.pos_encoder(spatial)
        spatial = self.spatial_TF(spatial)
        spatial = self.spatial_out_fc(spatial).permute(2, 1, 0)

        temporal = self.temporal_inp_fc(spatial) + step_temporal_embed
        temporal = self.pos_encoder(temporal)
        temporal = self.temporal_TF(temporal)
        temporal = self.temporal_out_fc(temporal)

        return temporal[x.shape[0]:]
# 将输入序列和噪声序列联合处理，通过空间个时间的transformer编码来进行降噪和预测


class Parallel_Denoiser(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len
        self.diff_steps = diff_steps

        self.spatial_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.spatial_TF = nn.TransformerEncoder(self.spatial_layer, num_layers=num_layers)
        self.spatial_inp_fc = nn.Linear(prefix_len+pred_len, qkv_dim)
        self.spatial_out_fc = nn.Linear(qkv_dim, prefix_len+pred_len)

        self.temporal_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.temporal_TF = nn.TransformerEncoder(self.temporal_layer, num_layers=num_layers)
        self.temporal_inp_fc = nn.Linear(input_dim, qkv_dim)
        self.temporal_out_fc = nn.Linear(qkv_dim, input_dim)

        self.combine_layer = nn.Conv2d(2, 1, 1, 1, 0)

        self.pos_encoder = PositionalEncoding(qkv_dim)
        
        self.step_temporal_encoder = DiffusionEmbedding(diff_steps, qkv_dim)
        self.step_spatial_encoder = DiffusionEmbedding(diff_steps, qkv_dim)

    def forward(self, noise, x, t):
        step_spatial_embed = self.step_spatial_encoder(t)
        step_temporal_embed = self.step_temporal_encoder(t)
        window = torch.cat([x, noise], dim=0)

        temporal = self.temporal_inp_fc(window) + step_spatial_embed
        temporal = self.pos_encoder(temporal)
        temporal = self.temporal_TF(temporal)

        spatial = self.spatial_inp_fc(window.permute(2, 1, 0)) + step_temporal_embed # L B D -> D B L 
        spatial = self.pos_encoder(spatial)
        spatial = self.spatial_TF(spatial)

        feat = torch.cat([self.temporal_out_fc(temporal).unsqueeze(1),
                          self.spatial_out_fc(spatial).permute(2, 1, 0).unsqueeze(1)], dim=1)
        out = self.combine_layer(feat).squeeze(1)
        return out[x.shape[0]:]

## codes referred from DLow's VAE code.
class GRUED_Denoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prefix_len = prefix_len
        self.pred_len = pred_len

        self.enc_gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers)
  
        self.dec_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers)

        self.step_encoder = DiffusionEmbedding(diff_steps, hidden_dim)

        self.inp_fc = nn.Linear(input_dim+hidden_dim, hidden_dim)
        self.init_h_fc = nn.Linear(hidden_dim, hidden_dim)
        self.init_y_fc = nn.Linear(hidden_dim, input_dim)
        self.out_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        feat, _ = self.enc_gru(x)
        return feat[-1]

    def decode(self, noise, feat, t):
        step_embed = self.step_encoder(t)

        h = self.init_h_fc(feat)

        rnn_in = torch.cat([noise, 
                            feat.unsqueeze(0).repeat(noise.shape[0], 1, 1)], dim=-1)
        rnn_in = self.inp_fc(rnn_in) + step_embed.unsqueeze(0)
        y, _ = self.dec_gru(rnn_in, h.unsqueeze(0).repeat(self.num_layers, 1, 1))
        res = self.out_fc(y)
        
        return res

    def forward(self, noise, x, t):
        return self.decode(noise, self.encode(x), t)

class GRU_Denoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prefix_len = prefix_len
        self.pred_len = pred_len

        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers)
  
        self.step_encoder = DiffusionEmbedding(diff_steps, hidden_dim)

        self.inp_fc = nn.Linear(input_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, noise, x, t):
        step_embed = self.step_encoder(t)
        window = torch.cat([x, noise], dim=0)

        out, _ = self.gru(self.inp_fc(window)+step_embed)
        out = self.out_fc(out)
        
        return out[x.shape[0]:]



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=2):
        super().__init__()
        self.proj = nn.Conv3d(46, 46, (1, 3, 3), padding=(0, 1, 1))  # 64  ,  64
        self.norm = nn.GroupNorm(groups, 46)  # 8,64
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim_out, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        x = rearrange(x, 'b c d -> b c d 1 1')
        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

def exists(x):
    return x is not None