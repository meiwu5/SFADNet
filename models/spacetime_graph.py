import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.device = torch.device("cuda:0")

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj
    
def generate_similarity_matrix(input_data):
    batch_size, num_features, num_nodes, node_dim = input_data.size()
    input_data = input_data.to("cuda:0")
    similarity_matrix = torch.matmul(input_data, input_data.transpose(2, 3)).to("cuda:0") 
    similarity_matrix = similarity_matrix.mean(dim=1).to("cuda:0")
    return similarity_matrix


class graph_constructor2(nn.Module):
    def __init__(self,  time_dim, k, dim, alpha=3, static_feat=None):
        super(graph_constructor2, self).__init__()
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(time_dim, dim)
            self.emb2 = nn.Embedding(time_dim, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.device = torch.device("cuda:0")
        self.adj_list = []

    def forward(self, idx,time_in_day_feat,day_in_week_feat ):
        self.adj_list = []
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1
            

        nodevec1 = torch.tanh(generate_similarity_matrix(time_in_day_feat))
        nodevec2 = torch.tanh(generate_similarity_matrix(day_in_week_feat))
        
        # nodevec1 的形状假设为 (batch_size, feature_dim)

        adj = torch.bmm(nodevec1, nodevec2.transpose(1,2))-torch.bmm(nodevec2, nodevec1.transpose(1,2)).to(self.device)
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask = mask.unsqueeze(0).expand(adj.size(0), -1, -1)
        mask_clone = mask.clone()  # 使用clone()方法克隆mask张量
        mask_clone.scatter_(1, t1, s1.fill_(1))
        adj = adj*mask_clone
#         print('时间邻接矩阵shape：',adj.shape)
#         adj = torch.stack(self.adj_list).to(self.device)
        return adj

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=1, dropout=True, bias=True):
        super().__init__()
        self.multi_head_self_attention  = MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout                    = nn.Dropout(dropout)

    def forward(self, X, K, V):
        hidden_states_MSA   = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA   = self.dropout(hidden_states_MSA)
        return hidden_states_MSA
    

class SpaceTime_graph(nn.Module):
    def __init__(self, nnodes, time_dim, k, dim, alpha, dropout, num_heads=1, bias=True):
        super().__init__()
        self.device = torch.device("cuda:0")  # 设置设备
        self.nnodes = nnodes
        self.time_dim = time_dim
        self.k = k
        self.dim = dim
        self.alpha = nn.Parameter(torch.tensor(3.0))
#         self.beta = nn.Parameter(torch.tensor(0.01))
        self.dropout = dropout
        self.num_nodes = torch.arange(self.nnodes).to(self.device)
        self.time_nodes = torch.arange(self.time_dim).to(self.device)
        self.transformer_layer = TransformerLayer(self.nnodes, num_heads, self.dropout, bias)
        self.gc = graph_constructor(self.nnodes, self.k, self.dim, alpha=self.alpha, static_feat=None)
        self.gc2 = graph_constructor2(self.time_dim, self.k, self.dim, alpha=self.alpha, static_feat=None)
        self.adj_list = []

    def forward(self,time_in_day_feat,day_in_week_feat):
        self.adj_list = []
        a1 = self.gc(self.num_nodes) #[N,N]
        a2 = self.gc2(self.time_nodes,time_in_day_feat,day_in_week_feat) #[B,N,N]
        a1 = a1.unsqueeze(0).expand(a2.size(0), -1, -1) #[B,N,N]
        a3 = F.relu(torch.tanh(self.alpha * torch.bmm(a1, a2))).to(self.device)#[B,N,N],就是相乘*可学习参数alpha
        mask = torch.zeros(self.num_nodes.size(0), self.num_nodes.size(0)).to(self.device)#掩码一下
        mask.fill_(float('0'))
        s1,t1 = (a3 + torch.rand_like(a3)*0.01).topk(self.k,1)
        mask = mask.unsqueeze(0).expand(a2.size(0), -1, -1)
        mask_clone = mask.clone()  # 使用clone()方法克隆mask张量
        mask_clone.scatter_(1, t1, s1.fill_(1))
        a3 = a3*mask_clone
        adj = self.transformer_layer(a1, a2, a3) #[B,N,N],[B,N,N],[B,N,N]
        return adj
    
    
class SpaceTime_graph2(nn.Module):
    def __init__(self, nnodes, time_dim, k, dim, alpha, dropout, num_heads=1, bias=True):
        super().__init__()
        self.device = torch.device("cuda:0")  # 设置设备
        self.nnodes = nnodes
        self.time_dim = time_dim
        self.k = k
        self.dim = dim
        self.alpha = nn.Parameter(torch.tensor(3.0))
#         self.beta = nn.Parameter(torch.tensor(0.01))
        self.dropout = dropout
        self.num_nodes = torch.arange(self.nnodes).to(self.device)
        self.time_nodes = torch.arange(self.time_dim).to(self.device)
        self.transformer_layer = TransformerLayer(self.nnodes, num_heads, self.dropout, bias)
        self.gc = graph_constructor(self.nnodes, self.k, self.dim, alpha=self.alpha, static_feat=None)
        self.gc2 = graph_constructor2(self.time_dim, self.k, self.dim, alpha=self.alpha, static_feat=None)
        self.adj_list = []

    def forward(self,time_in_day_feat,day_in_week_feat):
        self.adj_list = []
        a1 = self.gc(self.num_nodes) #[N,N]
        a2 = self.gc2(self.time_nodes,time_in_day_feat,day_in_week_feat) #[B,N,N]
        a1 = a1.unsqueeze(0).expand(a2.size(0), -1, -1) #[B,N,N]
        a3 = F.relu(torch.tanh(self.alpha * torch.bmm(a1, a2))).to(self.device)#[B,N,N],就是相乘*可学习参数alpha
        mask = torch.zeros(self.num_nodes.size(0), self.num_nodes.size(0)).to(self.device)#掩码一下
        mask.fill_(float('0'))
        s1,t1 = (a3 + torch.rand_like(a3)*0.01).topk(self.k,1)
        mask = mask.unsqueeze(0).expand(a2.size(0), -1, -1)
        mask_clone = mask.clone()  # 使用clone()方法克隆mask张量
        mask_clone.scatter_(1, t1, s1.fill_(1))
        a3 = a3*mask_clone
        adj = self.transformer_layer(a1, a2, a3) #[B,N,N],[B,N,N],[B,N,N]
        return adj