from basemodel import *
from GRU_decoder import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from AGT import *

class AGTFI(nn.Module):
    def __init__(self, args):
        super(AGTFI, self).__init__()
        self.args = args
        self.Temperal_Encoder = Temperal_Encoder(self.args)
        self.GRU_Dcoder = GRUDecoder(self.args)
        self.goal_decoder = Goal_decoder(self.args)
        self.GraphTransformer = GraphTransformer(self.args)
        

    def forward(self, inputs, edge_pair):
        
        batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split, max_values = inputs
        device = torch.device(batch_abs_gt.device)


        batch_abs_gt = batch_abs_gt[:self.args.obs_length,:,:2]
        self.batch_norm_gt = batch_norm_gt
        train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] 
        zeros = torch.zeros(1, train_x.size(1), 2, device=device)
        train_x = torch.cat([zeros, train_x], dim=0)

        train_x = train_x.permute(1, 2, 0) #[N, 2, H]
        train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0) #[N, 2, H]
        self.pre_obs=batch_norm_gt[1:self.args.obs_length] # 保存坐标
        
        x_encoded, hidden_state_unsplited = self.Temperal_Encoder.forward(train_x)  #[N, H, 2], [N, D], [N, D]
        batch_vol_gt = (batch_abs_gt[1:self.args.obs_length, :, :] - batch_abs_gt[:self.args.obs_length-1, :, :]).permute(1,2,0)[:, :, -1]
        batch_abs_gt = batch_abs_gt[self.args.obs_length-1, :, :] # 最后一个观测时刻的绝对坐标 [N, 2]
        batch_norm_abs_gt = (batch_abs_gt / max_values[0]).float() # 用最大坐标归一化坐标
        batch_norm_vol_gt = (batch_vol_gt / max_values[0]).float()
        

        hidden_state_global = hidden_state_unsplited

        for left, right in batch_split:
            now_pair = edge_pair[(left.item(), right.item())][0].to(device).long()
            hidden_now = hidden_state_unsplited[left: right].view(-1, self.args.hidden_size)
            edge_mat_list = []
            if len(now_pair) != 0:
                edge_pair_now = now_pair.transpose(0, 1)
                vol_now = batch_norm_vol_gt[left: right]
                x_norm_abs_now = batch_norm_abs_gt[left: right] 
                gt_out = self.GraphTransformer(hidden_now, x_norm_abs_now, edge_pair_now, vol_now)
                hidden_now = hidden_now + gt_out  
                hidden_state_global[left: right] = hidden_now

                
        
        train_y_gt = train_y.permute(0, 2, 1) # [N, H ,2]
        correction_features = self.goal_decoder(x_encoded.permute(1, 0, 2), hidden_state_global)
        loc_phase_2 = self.GRU_Dcoder(x_encoded, hidden_state_global, batch_split, correction_features)
        

        l2_loss, full_pre_tra = self.mdn_loss_nonly(train_y_gt, loc_phase_2)  #[K, H, N, 2]
        return l2_loss, full_pre_tra, loc_phase_2


    def mdn_loss_nonly(self, y, y_prime):
        batch_size=y.shape[0]
        y = y #[N, H, 2]
        pred_y = y_prime
        full_pre_tra = []
        l2_norm = (torch.norm(pred_y - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = pred_y[best_mode, torch.arange(batch_size)]
        l2_loss = torch.norm(y_hat_best-y,p=2,dim=-1).mean(-1).mean(-1)
        #best ADE
        sample_k = pred_y[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        # sample_k = out_mu.permute(1, 0, 2)
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        # best FDE
        l2_norm_FDE = (torch.norm(pred_y[:,:,-1,:] - y[:,-1,:], p=2, dim=-1) )  # [F, N]
        best_mode = l2_norm_FDE.argmin(dim=0)
        sample_k = pred_y[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        # if(torch.any(l2_loss == torch.nan)):
        #     print('erro')
        #     input()
        return l2_loss, full_pre_tra