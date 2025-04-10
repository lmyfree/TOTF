import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn


class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LatentMappingLayer, self).__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z = self.encode(x)
        return z

    def encode(self, x):
        h = self.enc1(x)
        # h = torch.dropout(h, 0.2, train=self.training)
        h = F.elu(h)
        h = self.enc2(h)
        h = F.elu(h)
        return h



class GraphEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, lam_emd=1., order=4):
        super(GraphEncoder, self).__init__()
        self.order = order
        self.SAL = GlobalSelfAttentionLayer(feat_dim, hidden_dim)
        self.lam_emd = lam_emd
        self.la = nn.Parameter(torch.ones(self.order))
        nn.init.normal_(self.la.data, mean=1., std=.001)
        # if cora, citeseer, pubmed
        # self.la = torch.ones(self.order)
        # self.la[1] = 2

    def forward(self, x, adj):
        # sattn = self.SAL(x)# + torch.eye(adj.shape[0], device=x.device)
        # a = [1. for i in range(self.order)]
        if self.order != 0:
            adj_temp = self.la[0] * adj.clone().detach()
            for i in range(self.order-1):
                adj_temp += self.la[i+1] * torch.matmul(adj, adj_temp).detach_()
            attn = adj_temp / self.order
            h2 = torch.mm(attn, x)
        else:
            h2 = x

        if True:  # or x.shape[0] not in [3327, 19717]:
            h1 = self.SAL(x)
        else:
            h1 = h2
       # h = torch.cat([h2, self.lam_emd * h1], dim=-1)
        h = torch.cat([h2, h2], dim=-1)
        return h


class GlobalSelfAttentionLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(GlobalSelfAttentionLayer, self).__init__()
        self.feat_dim = feat_dim
        #print(feat_dim.type)
        self.linear_query= nn.Linear(feat_dim, hidden_dim, bias=False)
        # self.K = nn.Linear(feat_dim, hidden_dim, bias=False)
        #self.Q = nn.Parameter(torch.zeros(feat_dim, hidden_dim))
        # self.Q = nn.Parameter(torch.zeros(1433, 256))
        # nn.init.xavier_normal_(self.Q.data, gain=1.141)
        self.linear_key = nn.Linear(feat_dim, hidden_dim, bias=False)
        # self.K = nn.Parameter(torch.zeros(1433, 256))
        # nn.init.xavier_normal_(self.K.data, gain=1.141)
        self.linear_values = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.sqrt_key_size = math.sqrt(feat_dim)


    def forward(self, common_z, zs, view_num):
        attn = []
        q = self.linear_query(common_z)#torch.matmul(common_z, self.Q)
        q = F.elu(q)
        for v in range(view_num):
          k = self.linear_key(zs[v])
          k = F.elu(k)

          att = torch.matmul(q, k.T)
          #a = F.softmax(att / self.sqrt_key_size, dim=0)
          att = F.normalize(att, p=2, dim=-1)
          attn.append(att)
        for i in range(view_num):
           print(attn[i].size())
           print(zs[i].size())
           common_z = common_z + torch.matmul(attn[i],zs[i])#attn[i] * zs[i]

       # h = torch.mm(attn, x)


        return common_z

