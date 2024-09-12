from torch.nn.parameter import Parameter
import torch
import torch.nn as nn   
import math

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
#         print('x.shape:',x.shape)
        x = torch.einsum('ncwl,nvw->ncvl',(x,G))
        return x    

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,nvw->ncvl',(x,A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class residualconv(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(residualconv, self).__init__()
        self.nconv = HGNN_conv(c_in,c_out)
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.a_list = []
    def forward(self,x,adj):
        matrix = torch.eye(adj.size(1)).to(x.device)
        matrix = matrix.unsqueeze(0)
        adj_new = adj + matrix
        d = adj_new.sum(1)
        d = d.view(adj_new.size(0), 1, adj_new.size(2))
        a = adj_new / d
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho
    
class residualconv2(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(residualconv2, self).__init__()
        self.nconv = HGNN_conv(c_in,c_out)
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.a_list = []
    def forward(self,x,adj):
        matrix = torch.eye(adj.size(1)).to(x.device)
        matrix = matrix.unsqueeze(0)
        adj_new = adj + matrix
        d = adj_new.sum(1)
        d = d.view(adj_new.size(0), 1, adj_new.size(2))
        a = adj_new / d
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho