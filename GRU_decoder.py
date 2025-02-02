
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from basemodel import MLP
from my_decoder import Conv_Decoder

import numpy as np



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, args, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.args = args
        self.d_model = d_model 
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pred_length = self.args.pred_length
        self.num_modes = self.args.final_mode
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # Final output projection
        self.mask_proj = nn.Sequential(
            nn.Linear(self.pred_length, self.num_modes*self.pred_length),
            nn.LayerNorm(self.num_modes *self.pred_length)
        )
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size() # x [20T, N, D]

        # x = self.pe(x)

        # Project input to queries, keys, and values
        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [20T, num_heads, N, heads_dim]
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5) #[20T, num_heads, N, N]
        attn_scores = torch.softmax(attn_scores, dim = -1)
        
        
        context = torch.matmul(attn_scores, values) 
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model) 
        output = self.out_projection(context)

        return output

class GRUDecoder(nn.Module):

    def __init__(self, args) -> None:
        super(GRUDecoder, self).__init__()
        # min_scale: float = 1e-3
        self.args = args
        self.n_ecoder = 1
        self.input_size = self.args.hidden_size
        self.hidden_size = self.args.hidden_size
        self.future_steps = self.args.pred_length

        self.num_modes = self.args.final_mode
        self.to12 = Conv_Decoder(args)
        self.self_attention_p = MultiHeadSelfAttention(self.args, self.hidden_size, 8)
        
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.multihead_proj_global = nn.Sequential(
                                    nn.Linear(self.input_size , self.num_modes * self.hidden_size), # self.num_modes = 20 ,实现多模态
                                    nn.LayerNorm(self.num_modes * self.hidden_size),
                                    nn.ReLU(inplace=True))
        self.GRU_encoders = GRU_Encoder_Block(self.args)
        self.Final_Decoder = GRU_Decoder_Block(self.args)

        self.multi_proj_goal = nn.Sequential(
                                    nn.Linear(self.hidden_size, self.num_modes * self.hidden_size),
                                    nn.LayerNorm(self.num_modes*self.hidden_size),
                                    nn.ReLU(inplace=True),
                                    )


    def forward(self, temporal_feature, interaction_feature, batch_split, goal): # [12, N, 64]  [N, 64]  [N, 64]  [N]  len(batch_split) = batch_szie
        his_tmp = self.multihead_proj_global(temporal_feature)\
            .view(-1, self.args.obs_length, self.num_modes, self.hidden_size)\
                .permute(1,2,0,3).contiguous()\
                    .view(self.args.obs_length,-1,self.hidden_size)      # [8,20N,D]
        temporal_feature = self.to12(his_tmp)
        his_social = interaction_feature.repeat(self.num_modes, 1, 1).reshape(1, -1, self.hidden_size) # [1, 20N, D]
        temporal_1 = temporal_feature
        refinement = torch.zeros_like(temporal_1)

        goal = self.multi_proj_goal(goal).view(-1, self.num_modes, self.hidden_size).transpose(0,1)

        temporal_1 = self.GRU_encoders(temporal_1, his_tmp[-1], his_social, goal) # [12, 20N, D]
        tmp = temporal_1.reshape(self.future_steps,self.num_modes,-1,self.hidden_size).transpose(0,2).reshape(-1,self.num_modes*self.future_steps,self.hidden_size) # [N,20T,D]
        tmp = tmp.detach() # [N,20T,D]  # (16)式
        refinement = refinement.reshape(-1,self.num_modes*self.future_steps,self.hidden_size) # [N,20T,D]
        for left, right in batch_split:
            out_p = self.self_attention_p(tmp[left:right].transpose(0,1))
            refinement[left:right] = out_p.transpose(0,1)
        refinement = refinement.view(-1,self.num_modes,self.future_steps,self.hidden_size).transpose(0,2).reshape(self.future_steps,-1,self.hidden_size) # [12,20N,D]   # 二次交互后的特征


        h_lstm2 = self.Final_Decoder(temporal_feature, his_tmp[-1], his_social, goal, refinement)
        output2 = h_lstm2
        loc_phase_2 = self.loc(output2).view(self.num_modes, -1, self.future_steps, 2) # [20, N, 12, 2]  # (34)式


        return loc_phase_2 


                
class Gated_fuse(nn.Module):
    def __init__(self,args):
        super(Gated_fuse, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        self.encoder_weight = nn.Sequential(
            nn.Linear(self.hidden_size*2,self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Sigmoid()
        )

    def forward(self,past_feat,dest_feat):
        fuse =torch.cat((past_feat,dest_feat),dim=-1)
        weight = self.encoder_weight(fuse)
        fused = past_feat*weight + dest_feat*(1-weight)

        return fused
    

class GRU_Encoder_Block(nn.Module):
    def __init__(self,args):
        super(GRU_Encoder_Block, self).__init__()
        self.args = args
        self.num_modes = self.args.final_mode
        self.future_steps = args.pred_length
        self.hidden_size = self.args.hidden_size
        self.GRU_cell_mid = nn.GRUCell(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    bias=True)
        self.mlp_for_layer = nn.Sequential(
            nn.Linear(self.hidden_size,2*self.hidden_size),
            nn.LayerNorm(2*self.hidden_size),
            nn.Linear(2*self.hidden_size,self.hidden_size),
            nn.LayerNorm(self.hidden_size),            
            nn.ReLU()
        )
        self.GRU_cell_for = nn.GRUCell(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    bias=True)
        self.mlp_bac_layer = nn.Sequential(
            nn.Linear(self.hidden_size,2*self.hidden_size),
            nn.LayerNorm(2*self.hidden_size),
            nn.Linear(2*self.hidden_size,self.hidden_size),
            nn.LayerNorm(self.hidden_size),            
            nn.ReLU()
        )
        self.GRU_cell_backward = nn.GRUCell(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    bias=True)
        self.goal_h = nn.Sequential(
            MLP(self.hidden_size,self.hidden_size),
        )
        self.fuse = Gated_fuse(self.args)
    
    def forward(self, temporal, h_temp, h, goal):
        h_mid = h.squeeze(0)
        out_mid_list = []
        for t in range(self.future_steps):
            h_mid = self.GRU_cell_mid(temporal[t], h_mid) #
            out_mid_list.append(h_mid)
        v_mid = torch.stack(out_mid_list)

        v_for = h + self.mlp_for_layer(v_mid)
        output_h_for = []
        h_for = h_temp
        for t in range(self.future_steps):
            h_for = self.GRU_cell_for(v_for[t], h_for)     # GRU CELL
            output_h_for.append(h_for)
        output_h_for = torch.stack(output_h_for, dim=0)

        v_back = h + self.mlp_bac_layer(v_mid)
        output_h_back = []
        h_back = self.goal_h(goal).reshape(-1,self.hidden_size)
        for t in range(self.future_steps-1,-1,-1):
            h_back = self.GRU_cell_backward(v_back[t],h_back)      
            output_h_back.append(h_back)
        output_h_back = torch.stack(output_h_back, dim=0)         
        output_h_back_flip = torch.flip(output_h_back, dims=[0])
        output_h_for = output_h_for.transpose(0, 1)
        output_h_back_flip = output_h_back_flip.transpose(0, 1)
        out = self.fuse(output_h_for,output_h_back_flip)

        return out.transpose(0,1)
        


class GRU_Decoder_Block(nn.Module):
    def __init__(self,args):
        super(GRU_Decoder_Block, self).__init__()
        self.args = args
        self.num_modes = self.args.final_mode
        self.future_steps = args.pred_length
        self.hidden_size = self.args.hidden_size
        self.GRU_cell_mid = nn.GRUCell(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    bias=True)
        self.mlp_for_layer = nn.Sequential(
            nn.Linear(self.hidden_size,2*self.hidden_size),
            nn.LayerNorm(2*self.hidden_size),
            nn.Linear(2*self.hidden_size,self.hidden_size),
            nn.LayerNorm(self.hidden_size),            
            nn.ReLU()
        )
        self.GRU_cell_for = nn.GRUCell(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    bias=True)
        self.mlp_bac_layer = nn.Sequential(
            nn.Linear(self.hidden_size,2*self.hidden_size),
            nn.LayerNorm(2*self.hidden_size),
            nn.Linear(2*self.hidden_size,self.hidden_size),
            nn.LayerNorm(self.hidden_size),            
            nn.ReLU()
        )
        self.GRU_cell_backward = nn.GRUCell(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    bias=True)
        self.goal_h = nn.Sequential(
            MLP(self.hidden_size,self.hidden_size),
        )
        self.fuse = Gated_fuse(self.args)
    
    def forward(self, temporal, h_temp, h, goal, refinement):
        h_mid = h.squeeze(0)
        out_mid_list = []
        for t in range(self.future_steps):
            h_mid = h_mid + refinement[t]
            h_mid = self.GRU_cell_mid(temporal[t], h_mid) 
            out_mid_list.append(h_mid)
        v_mid = torch.stack(out_mid_list)

        v_for = h + self.mlp_for_layer(v_mid)
        output_h_for = []
        h_for = h_temp
        for t in range(self.future_steps):
            h_for = self.GRU_cell_for(v_for[t], h_for) 
            output_h_for.append(h_for)
        output_h_for = torch.stack(output_h_for, dim=0)

        v_back = h + self.mlp_bac_layer(v_mid)
        output_h_back = []
        h_back = self.goal_h(goal).reshape(-1,self.hidden_size)
        for t in range(self.future_steps-1,-1,-1): 
            h_back = self.GRU_cell_backward(v_back[t],h_back)    
            output_h_back.append(h_back)
        output_h_back = torch.stack(output_h_back, dim=0)          
        output_h_back_flip = torch.flip(output_h_back, dims=[0])
        output_h_for = output_h_for.transpose(0, 1)
        output_h_back_flip = output_h_back_flip.transpose(0, 1)
        out = self.fuse(output_h_for,output_h_back_flip)

        return out
    


