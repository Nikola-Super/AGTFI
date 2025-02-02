import torch
from torch import nn
import torch.nn.functional as F

class Conv_Decoder(nn.Module):

    def __init__(self, args):
        super(Conv_Decoder, self).__init__()

        self.hidden_size = args.hidden_size
        self.n_tcns = 6
        self.dropout = 0.1
        self.args = args


        # 接受的输入为[T, N, C]
        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(
            nn.Conv2d(self.args.obs_length, self.args.pred_length, (1, 3), padding=(0, 1)), # 通道注意力卷积
            nn.PReLU()
        ))
        for j in range(1, self.n_tcns):
            self.tcns.append(nn.Sequential(
                nn.Conv2d(self.args.pred_length, self.args.pred_length, (1, 3), padding=(0, 1)),
                nn.PReLU()
        ))

    def forward(self,x_encoded):
        tcn_out = self.tcns[0](x_encoded)
        for k in range(1, self.n_tcns):
            tcn_out = F.dropout(self.tcns[k](tcn_out) + tcn_out, p=self.dropout)
        

        return tcn_out