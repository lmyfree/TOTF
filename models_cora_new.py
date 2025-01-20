import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
from IGAE1 import IGAE_encoder, IGAE_decoder
from layers import GraphEncoder, LatentMappingLayer,GlobalSelfAttentionLayer
from sklearn.cluster import KMeans


class commonExtractor(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, class_num, lam_emd=1., alpha=0.2, order=5, view_num=2):
        super(commonExtractor, self).__init__()
        self.hidden_size = hidden_dim
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.class_num = class_num

        #self.GlobalSelfAttentionLayer = GlobalSelfAttentionLayer(64, 64)
        self.cluster_layer = [Parameter(torch.Tensor(class_num, latent_dim)) for _ in range(view_num)]
        self.cluster_layer.append(Parameter(torch.Tensor(class_num, view_num * latent_dim)))
        self.LatentMap = LatentMappingLayer(128, hidden_dim, 64)
        # self.cluster_layer.append(torch.cat(self.cluster_layer, dim=-1))
        self.GraphEnc = [IGAE_encoder(gae_n_enc_1=128, gae_n_enc_2=256, gae_n_enc_3=64,n_input=feat_dim[k]) for k in range(view_num)]# 128, 256,64
        #yuan-self.GraphEnc = [GraphEncoder(feat_dim[k], hidden_dim, lam_emd=lam_emd, order=order) for k in range(view_num)]
       # self.LatentMap = [LatentMappingLayer(64, hidden_dim, 64) for t in range(view_num)]#[LatentMappingLayer(2*feat_dim[t], hidden_dim, latent_dim) for t in range(view_num)]
        #self.FeatDec = [LatentMappingLayer(latent_dim, hidden_dim, feat_dim[e]) for e in range(view_num)]
        #self.FeatDec = [IGAE_decoder(gae_n_dec_1=64, gae_n_dec_2=256, gae_n_dec_3=128,n_input=feat_dim[e]) for e in range(view_num)]

        for i in range(view_num):
            self.register_parameter('centroid_{}'.format(i), self.cluster_layer[i])
            self.add_module('graphenc_{}'.format(i), self.GraphEnc[i])
            self.add_module('latentmap_{}'.format(i), self.LatentMap)
          #  self.add_module('featdec_{}'.format(i), self.FeatDec[i])
        self.register_parameter('centroid_{}'.format(view_num), self.cluster_layer[view_num])

    def forward(self, views, zs, qs, view_num):
        # x = torch.dropout(x, 0.2, train=self.training)
        #e = self.GraphEnc[view](x, adj)
        # print('view:', view, self.GraphEnc[view].la)
        # print(self.cluster_layer[view])
        for view in range(view_num):
            adj, X = views[view]
            #x = torch.tensor(features[view])#.cuda()
            z = self.GraphEnc[view](X, adj)
            zs.append(z)

            #z = self.LatentMap[view](z)
           # z = torch.tensor(z)
            z_norm = F.normalize(z, p=2, dim=1)
          #  A_pred = self.decode(z_norm)
            q = self.predict_distribution(z_norm, view)

            qs.append(q)
        chushi_z = torch.cat([zs[i] for i in range(view_num)], dim=-1)
        common_z = self.LatentMap(chushi_z)
       # common_z = self.GlobalSelfAttentionLayer(common_z, zs, view_num)
        return zs, qs, common_z

    @staticmethod
    def decode(z):
        rec_graph = torch.sigmoid(torch.matmul(z, z.T))
        return rec_graph

    def predict_distribution(self, z, v, alpha=1.0):
        km = KMeans(7).fit(z.detach().numpy())
        # G = km.labels_
        c = km.cluster_centers_
       # c = self.cluster_layer[v]
        # print("c", c)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.detach().unsqueeze(1) - c, 2), 2) / alpha)
        # print("q0", q)
        q = q.pow((alpha + 1.0) / 2.0)
        # print("q1", q)
        q = (q.t() / torch.sum(q, 1)).t()
        q.requires_grad = True
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()


    def get_graph_embedding(self, x, adj,view):
        e = self.GraphEnc[view](x, adj)
        e_norm = F.normalize(e, p=2, dim=1)

        return e_norm

