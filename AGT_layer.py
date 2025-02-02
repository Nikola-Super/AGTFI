import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer with edge features
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func

def save_for_heat_map(edge_feat):
    def func(edges):
        return {'e_for_heat': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5e-3,5e-3))}
    return func

def edge_gate_feat(src_field, dst_field, edge_feat, out_field):
    def func(edges):
        return {out_field: torch.cat((edges.src[src_field], edges.dst[dst_field], edges.data[edge_feat]), dim=-1)}
    return func

def apply_gate(attention_score, gate):
    def func(edges):
        return {gate: torch.mul(edges.data[attention_score], torch.sigmoid(edges.data[gate]))}
    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = self.out_dim // self.num_heads
        
        self.change_channel = nn.Sequential(
            nn.Linear(3 * out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh() # 这个Tanh能删掉吗？
        )

        self.conv1d = nn.Sequential(conv_channels(self.num_heads, self.out_dim),
                                    conv_channels(self.num_heads, self.out_dim)
                                    )
        self.sigmoid = nn.Sigmoid()

        self.Q = nn.Linear(in_dim, out_dim * num_heads)
        self.K = nn.Linear(in_dim, out_dim * num_heads)
        self.V = nn.Linear(in_dim, out_dim * num_heads)
        self.proj_e = nn.Linear(in_dim, out_dim * num_heads)
        self.n_gate = nn.Sequential(
            nn.Linear(in_dim, out_dim * num_heads // 2),
            #nn.ReLU(),
            nn.LayerNorm(out_dim* num_heads//2), # 这里能删掉吗？
            nn.Tanh(),
            nn.Linear(out_dim * num_heads // 2, out_dim* num_heads)
        )
    
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges) # 点对相乘获得'score'保存在边上
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e')) # score与边原有的信息'proj_e'相乘，并保存在'score'中

        # 点对特征和要用的proj_e特征cat到一起，经过mlp给e_gate
        g.apply_edges(edge_gate_feat('K_h', 'Q_h', 'proj_e', 'e_gate'))

        g.edata['e_gate'] = self.change_channel(g.edata['e_gate'])
      
        g.edata['e_gate'] = self.conv1d(g.edata['e_gate'])
        # g.edata['e_gate'] = self.sigmoid(g.edata['e_gate'])
       
        g.apply_edges(out_edge_features('score')) # 将'score'信息保存在'e_out'中
        g.apply_edges(apply_gate('score', 'e_gate'))
        g.apply_edges(save_for_heat_map('e_gate')) # 保存热度数据到e_for_heat
        # Copy edge features as e_out to be passed to FFN_e
        
        
        # softmax
        g.apply_edges(exp('e_gate')) # softmax, 范围在(-5,5)之间

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'e_gate', 'V_h'), fn.sum('V_h', 'wV')) 
        # 结点的消息传递 
        # dgl.function.scr_mul_edge:将源节点的特征与边的分数相乘。
        # dgl.function.sum('V_h','wV'):将'V_h'沿着边进行求和，然后将求和结果保存到目标节点的特征'wV'中
        # g.send_and_recv:沿着边的id将消息从源节点传递到目标节点。
        g.apply_nodes(lambda nodes: {'wV': torch.mul(nodes.data['wV'], nodes.data['n_gate'])})
        g.send_and_recv(eids, fn.copy_edge('e_gate', 'e_gate'), fn.sum('e_gate', 'z'))
    
    def forward(self, g, h, e):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        n_gate = self.n_gate(h)
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)
        g.ndata['n_gate'] = torch.sigmoid(n_gate.view(-1, self.num_heads, self.out_dim))

        self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-30)) # adding eps to all values here
        e_out = g.edata['e_out']
        

        
        
        return h_out, e_out 
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, args, in_dim, out_dim, num_heads, dropout=0.0):
        super().__init__()

        self.args = args
        if self.args.phase == 'train':
            self.training = True
        else:
            self.training = False

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dropout_layer_h = nn.Dropout(p=self.dropout)
        self.dropout_layer_e = nn.Dropout(p=self.dropout)
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads)
        
        # # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(2*out_dim, out_dim)
        
        # # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(2*out_dim, out_dim)

    
        self.layer_norm_h = nn.LayerNorm(out_dim)
        self.layer_norm_e = nn.LayerNorm(out_dim)
            
        
        self.NodeRes_1 = LearnableCoefficient()
        self.NodeRes_2 = LearnableCoefficient()
        self.EdgeRes_1 = LearnableCoefficient()
        self.EdgeRes_2 = LearnableCoefficient()

        
    def forward(self, g, h, e):
        h_res1 = h # for first residual connection
        e_res1 = e # for first residual connection
        
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e) # 消息传递
        
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)
    
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        if self.training:
            h = self.dropout_layer_h(h)
        h = self.FFN_h_layer2(h)

        
        h = self.NodeRes_1(h_res1) + self.NodeRes_2(h)  # residual connection       
        e = self.EdgeRes_1(e_res1) + self.NodeRes_2(e) # residual connection  

        h = self.layer_norm_h(h)
        e = self.layer_norm_e(e)


        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads)
    

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        # print(self.bias)
        return out
    
    
class conv_channels(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.conv = nn.Sequential(nn.Conv1d(in_channels=self.num_heads, out_channels=self.num_heads, kernel_size=3,padding=1)
        )

    def forward(self, x):
        conved_x = self.conv(x)
        return conved_x