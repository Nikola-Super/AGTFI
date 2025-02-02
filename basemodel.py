
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 




def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) :
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            # print("LSTM------",m.named_parameters())
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)  # initializing the lstm bias with zeros
        else:
            print(m,"************")



class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
        
class MLP_gate(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP_gate, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return hidden_states

class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states
        

class Temperal_Encoder(nn.Module):
    """Construct the sequence model"""

    def __init__(self,args):
        super(Temperal_Encoder, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        # if args.input_mix:
        #     self.conv1d=nn.Conv1d(4, self.hidden_size, kernel_size=3, stride=1, padding=1)
        # else:
        self.conv1d=nn.Conv1d(2, self.hidden_size, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead= self.args.x_encoder_head,\
                         dim_feedforward=self.hidden_size, batch_first=True) # 每个agent看作一个batch
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.args.x_encoder_layers)
        self.mlp1 = MLP(self.hidden_size)
        self.mlp = MLP(self.hidden_size)
        self.GRU = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=False)
        initialize_weights(self.conv1d.modules())

    def forward(self, x): # [N, 2, H]
        self.x_dense=self.conv1d(x).permute(0,2,1) #[N, H, dim]
        self.x_dense=self.mlp1(self.x_dense) + self.x_dense #[N, H, dim]
        self.x_dense_in = self.transformer_encoder(self.x_dense) + self.x_dense  #[N, H, D]
        output, hn = self.GRU(self.x_dense_in) # output.shape = [N, H, D]
        self.x_state = hn.squeeze(0) #[N, D] 
        self.x_endoced=self.mlp(output) + output # [N, H, D]
        return self.x_endoced, self.x_state


class Goal_decoder(nn.Module):
    def __init__(self, args):
        super(Goal_decoder,self).__init__()
        #self.porj()
        self.args = args
        self.MLP = nn.Linear(2*self.args.hidden_size, 1)
        self.conv2d = nn.Sequential(
            nn.Conv2d(self.args.obs_length, 1, (1, 3), padding=(0, 1)),
            nn.PReLU()
            )
    def forward(self, x_ecoded, hidden_state_global):
        tmp_encoded = self.conv2d(x_ecoded).squeeze(0) # x: [T, N, D] [N,D]
        return tmp_encoded
        