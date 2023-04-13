import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os
from utils import FourierFeatureTransform
from utils import device
from torch import nn
import torch
from einops.einops import rearrange, repeat
import numpy as np
import math 

class ParamDecoder(nn.Module):
    def __init__(self, mu_dim, need_in_dim,need_out_dim,k=30):
        super(ParamDecoder, self).__init__()
        self.need_in_dim=need_in_dim
        self.need_out_dim=need_out_dim
        self.k=k
        self.decoder = nn.Linear(mu_dim, need_in_dim*k) 
        self.V = nn.parameter.Parameter(torch.zeros(k,need_out_dim))
      
    def forward(self, t_feat):
        B=t_feat.shape[0]
        U = self.decoder(t_feat).reshape(B,self.need_in_dim,self.k)  # B x need_in_dim x k
        param=torch.einsum('bik,kj->bij',U,self.V).reshape(B,-1)
        return param

class DynamicLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mu_dim: int, bias=True):
        super(DynamicLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mu_dim = mu_dim
        self.bias=bias
        self.decoder = ParamDecoder(mu_dim, in_dim + 1, out_dim)
    def forward(self, x, mu):

        param=rearrange(self.decoder(mu),'B (dim_A dim_B) -> B dim_A dim_B',dim_A=self.in_dim+1,dim_B=self.out_dim)
        weight=param[:,:-1,:]
        bias=param[:, -1, :]
        x=torch.einsum('b...d,bde->b...e',x,weight)
        if self.bias:
            bias=bias.view(((bias.shape[0],)+(1,)*(len(x.size())-2)+(bias.shape[-1],)))
            x=x+bias
        return x
    
class MuModuleList(nn.ModuleList):
    def forward(self,x,mu):
        for layer in self:
            if type(layer) == DynamicLinear:
                x=layer(x,mu)
            else:
                x=layer(x)
        return x
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels,text_dim, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = MuModuleList([
            DynamicLinear(gate_channels, gate_channels // reduction_ratio,text_dim),
            nn.ReLU(),
            DynamicLinear(gate_channels // reduction_ratio, gate_channels,text_dim)
        ])
        self.pool_types = pool_types
    def forward(self, x ,mu):
        dynamic_x = self.mlp(x ,mu).mean(dim=0,keepdim=True) #1,D
        scale = torch.sigmoid( dynamic_x )
        res = x * scale
        return res


class SpatialGate(nn.Module):
    def __init__(self,gate_channels,mu_dim, reduction_ratio=16, pool_types=['avg', 'max']):
        super(SpatialGate, self).__init__()
        self.pool_types = pool_types
        self.mlp =  MuModuleList([
            DynamicLinear(gate_channels, gate_channels // reduction_ratio,mu_dim),
            nn.ReLU(),
            DynamicLinear(gate_channels // reduction_ratio, gate_channels,mu_dim)
        ])
    def forward(self, x, mu):
        dynamic_x = self.mlp(x ,mu).mean(dim=1,keepdim=True) #P,1
        scale = torch.sigmoid(dynamic_x) 
        res=x*scale # broadcasting (P,D)
        return res

class QueryDynamicAttention(nn.Module):
    def __init__(self,gate_channels=256,mu_dim=512, reduction_ratio=8, pool_types=['avg', 'max'],use_spatial=True,use_channel=True):
        super(QueryDynamicAttention,self).__init__()
        self.ChannelGate = ChannelGate(gate_channels,mu_dim, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(gate_channels,mu_dim, reduction_ratio, pool_types)
        self.use_spatial=use_spatial
        self.use_channel=use_channel
    def forward(self, x,mu):
        if self.use_channel:
            x = self.ChannelGate(x,mu) 
        if self.use_spatial:
            x = self.SpatialGate(x,mu) 
        return x

class ProgressiveEncoding(nn.Module):
    def __init__(self, mapping_size, T, d=3, apply=True):
        super(ProgressiveEncoding, self).__init__()
        self._t = 0
        self.n = mapping_size
        self.T = T
        self.d = d
        self._tau = 2 * self.n / self.T
        self.indices = torch.tensor([i for i in range(self.n)], device=device)
        self.apply = apply
    def forward(self, x):
        alpha = ((self._t - self._tau * self.indices) / self._tau).clamp(0, 1).repeat(
            2)  # no need to reduce d or to check cases
        if not self.apply:
            alpha = torch.ones_like(alpha, device=device)  ## this layer means pure ffn without progress.
        alpha = torch.cat([torch.ones(self.d, device=device), alpha], dim=0)
        self._t += 1
        return x * alpha


class NeuralStyleField(nn.Module):
    # Same base then split into two separate modules 
    def __init__(self, sigma, depth, width, encoding, colordepth=2, normdepth=2, normratio=0.1, clamp=None,
                 normclamp=None,niter=6000, input_dim=3, progressive_encoding=True, exclude=0):
        super(NeuralStyleField, self).__init__()
        self.pe = ProgressiveEncoding(mapping_size=width, T=niter, d=input_dim)
        self.clamp = clamp
        self.normclamp = normclamp
        self.normratio = normratio
        self.dynamicAttention = QueryDynamicAttention()
        self.ln_appear = nn.LayerNorm(width)
        self.ln_shape = nn.LayerNorm(width)
        

        layers = []
        if encoding == 'gaussian':
            layers.append(FourierFeatureTransform(input_dim, width, sigma, exclude))
            if progressive_encoding:
                layers.append(self.pe)
            layers.append(nn.Linear(width * 2 + input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(width))
            
        else:
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(width))
            
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(width))
        self.base = nn.ModuleList(layers)

        # Branches 
        color_layers = []
        for _ in range(colordepth):
            color_layers.append(nn.Linear(width, width))
            color_layers.append(nn.ReLU())
            color_layers.append(nn.LayerNorm(width))
        color_layers.append(nn.Linear(width, 3))
        self.mlp_rgb = nn.ModuleList(color_layers)

        normal_layers = []
        for _ in range(normdepth):
            normal_layers.append(nn.Linear(width, width))
            normal_layers.append(nn.ReLU())
            normal_layers.append(nn.LayerNorm(width))
        normal_layers.append(nn.Linear(width, 3))
        self.mlp_normal = nn.ModuleList(normal_layers)
        print(self.base)
        print(self.mlp_rgb)
        print(self.mlp_normal)

    def reset_weights(self):
        self.mlp_rgb[-1].weight.data.zero_()
        self.mlp_rgb[-1].bias.data.zero_()
        self.mlp_normal[-1].weight.data.zero_()
        self.mlp_normal[-1].bias.data.zero_()

    def forward(self, x,prompt):
        for layer in self.base:
            x = layer(x) # points, dim
        x = self.dynamicAttention(x,prompt.float())
        
        colors = self.mlp_rgb[0](x)
        for layer in self.mlp_rgb[1:]:
            colors = layer(colors)
        displ = self.mlp_normal[0](x)
        for layer in self.mlp_normal[1:]:
            displ = layer(displ)

        if self.clamp == "tanh":
            colors = F.tanh(colors) / 2
        elif self.clamp == "clamp":
            colors = torch.clamp(colors, 0, 1)
        if self.normclamp == "tanh":
            displ = F.tanh(displ) * self.normratio / np.sqrt(3)
        elif self.normclamp == "clamp":
            displ = torch.clamp(displ, -self.normratio, self.normratio) / np.sqrt(3)

        return colors, displ



def save_model(model, loss, iter, optim, output_dir):
    save_dict = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss
    }

    path = os.path.join(output_dir, 'checkpoint.pth.tar')

    torch.save(save_dict, path)


