from scipy import io
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize
import torch
from sklearn.feature_extraction.text import TfidfTransformer
import os
import scipy.io as sio
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
#from ogb.nodeproppred import NodePropPredDataset, Evaluator
from sklearn.neighbors import kneighbors_graph
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def Aminer():
  data = sio.loadmat("/home/hnu/Disk0/lmy/dataset/Aminer_processed.mat")
  #dataset = "data/ACM3025"
  #data = io.loadmat('{}.mat'.format(dataset))
  C = data["PvsA"].toarray()
  # [123:223,200:300]
  B = data["AvsA"].toarray()
  A = C.T @ C
  # P=A
  A[A != 0] = 1
  X = data["AvsF"]
  #labels = data["AvsC"].T
  labels = np.argmax(data["AvsC"],axis=1).reshape(-1)
  labels = labels.T
  labels = labels.A1


  Xs = []
  As = []
  # count = np.count_nonzero(B == 1)
  # print(count)

 # Xs.append(X)
  Xs.append(X)
  As.append(A)
  As.append(B)


  #labels = labels.reshape(-1)

  return As, Xs, labels
def acm():
  dataset = "data/ACM3025"
  data = io.loadmat('{}.mat'.format(dataset))

  X = data['feature']
  A = data['PAP']
  B = data['PLP']

  Xs = []
  As = []

 # Xs.append(X)
  Xs.append(X)
  As.append(A)
  As.append(B)

  labels = data['label']
  labels = labels.T
  labels = np.argmax(labels, axis=0)
  #labels = labels.reshape(-1)

  return As, Xs, labels

def dblp():
  dataset = "/home/hnu/Disk0/lmy/MAGC/DBLP4057_GAT_with_idx"
  data = io.loadmat('{}.mat'.format(dataset))

  X = data['features']
 # A = data['net_APTPA']
  B = data['net_APCPA']
  C = data['net_APA']

  Xs = []
  As = []

  Xs.append(X)
 # As.append(A)
  As.append(B)
  As.append(C)

  labels = data['label']
  labels = labels.T
  labels = np.argmax(labels, axis=0)

  return As, Xs, labels
def mark_rows_with_zero(matrix):
    # 创建一个布尔掩码，检查每一行是否有0
  #  mask = (matrix == 0).any(dim=1)  # 生成的布尔掩码，标记有0的行

    # 创建一个标记矩阵
    # 使用float类型，这样可以标记为0或1
    marked_matrix = torch.zeros(matrix.size(0),64, dtype=torch.float)
    marked_matrixt = torch.randint(0, 2, (matrix.size(0), 64))
  #  marked_matrix[mask,:] = 0  # 将包含0的行标记为0
   # marked_matrix[~mask,:] = 1  # 将不包含0的行标记为1(2708)
   # masked_matrix = masked_matrix.numpy()
    return marked_matrix


def zero_out_rows_with_zero(matrix):
    # 创建一个布尔掩码，检查每一行是否有0
    marked_matrixt = torch.zeros(matrix.size(0), matrix.size(1))
    #random_tensor = torch.rand(matrix.size(0),matrix.size(1))  # 生成指定形状的随机数
   # masked_matrixt = (random_tensor > 0.8).int()
   # mask = (matrix == 0).any(dim=1)  # 生成的布尔掩码，标记有0的行
    marked_matrixt = torch.randint(0, 2, (matrix.size(0),matrix.size(1)))
    #marked_matrixt = torch.zeros(matrix.size(0),matrix.size(1), dtype=torch.float)
    # 将包含零的行的所有元素设置为0
   # marked_matrixt[mask] = torch.randint(0, 2, mask.size())

    return marked_matrixt

def imdb():
  dataset = "/home/hnu/Disk0/lmy/MAGC/imdb5k"
  data = io.loadmat('{}.mat'.format(dataset))
  
  X = data['feature']
  A = data['MAM']
  B = data['MDM']
  
  Xs = []
  As = []

  Xs.append(X)
  As.append(A)
  As.append(B)

  #labels = data['label']
  #labels = labels.reshape(-1)
  labels = data['label']
  labels = labels.T
  labels = np.argmax(labels, axis=0)

  return As, Xs, labels


def photos():
  dataset = 'Amazon_photos'
  data = io.loadmat('{}.mat'.format(dataset))
  
  X = data['features'].toarray().astype(float)
  A = data.get('adj')
  labels = data['label']
  labels = labels.reshape(-1)
  
  As = [A, A]
  
  Xs = [X, np.log2(1+X)]

  return As, Xs, labels
  
  

def wiki():
  data = io.loadmat(os.path.join('', f'wiki.mat'))
  X = data['fea'].toarray().astype(float)
  A = data.get('W')#.toarray()
  labels = data['gnd'].reshape(-1)
  
  As = [A, kneighbors_graph(X, 5, metric='cosine')]
  Xs = [X, np.log2(1+X)]

  return As, Xs, labels
def BlogCatalog():
    dataset = 'BlogCatalog'
    data = io.loadmat('{}.mat'.format(dataset))

    X = data['Attributes'].toarray().astype(float)
    A = data['Network'].toarray()#.get('W')
    #X=np.vstack([X, X])
   # X=np.repeat(X,2, axis=0)
    labels = data['Label']
    labels = labels.reshape(-1)
   # labels = np.repeat(labels, 2, axis=0)
    #labels = np.hstack([labels,labels])
   # A = np.repeat(A, 2, axis=0)
   # A = np.repeat(A, 2, axis=1)
    As = [A]

    Xs = [X, X @ X.T]

    return As, Xs, labels
def Flickr():
    dataset = '/home/hnu/Disk0/lmy/MAGC/Flickr'
    data = io.loadmat('{}.mat'.format(dataset))

    X = data['Attributes'].toarray().astype(float)
    A = data['Network'].toarray()#.get('W')
    labels = data['Label']
    labels = labels.reshape(-1)

    As = [A]

    Xs = [X, X @ X.T]

    return As, Xs, labels

def arxiv():
    dataset = 'arxiv'
    #data = io.loadmat('{}.mat'.format(dataset))
    #dataset = NodePropPredDataset(name=f'ogbn-arxiv')
   # data = dataset[0]
   # print(data)
    #adj = sp.coo_matrix(
    #    (np.ones(1166243), (data[0]["edge_index"][0], data[0]["edge_index"][1])),
    #    shape=(169343, 169343))

    #X = data['Attributes'].toarray().astype(float)
    # = data[0]["node_feat"]#.toarray()#.get('W')
    #labels = data[1]
    #labels = labels.reshape(-1)
    #del data, dataset
   # adj=adj.toarray()
    f = open('adj1.pkl','rb')
    adj = pickle.load(f)
    p = open('X1.pkl','rb')
    X = pickle.load(p)
    R = open('labels1.pkl','rb')
    labels = pickle.load(R)
   # labels = labels[:100000]
   # adj = adj.tocsr()[:100000, :100000]
   # adj = adj.tocsr()
   # adj = adj.toarray()
   # X = [X[:100000, :]]
    As = [adj]
    Xs = [X,X]
    # X1 = X[:100000, :]
    # print(len(As))
    # print(len(Xs))
    # print("打印AS")
    # print(As)
    # print("打印Xs的shape")
    # print(X1.shape)
    del adj, X
    return As, Xs, labels
def IMDB1():
    dataset = 'IMDB1'
    #data = io.loadmat('{}.mat'.format(dataset))
    #dataset = NodePropPredDataset(name=f'ogbn-arxiv')
   # data = dataset[0]
   # print(data)
    #adj = sp.coo_matrix(
    #    (np.ones(1166243), (data[0]["edge_index"][0], data[0]["edge_index"][1])),
    #    shape=(169343, 169343))

    #X = data['Attributes'].toarray().astype(float)
    # = data[0]["node_feat"]#.toarray()#.get('W')
    #labels = data[1]
    #labels = labels.reshape(-1)
    #del data, dataset
   # adj=adj.toarray()
    f = open('/home/hnu/Disk0/lmy/MAGC/imdb-f.pkl','rb')
    X = pickle.load(f)
    p = open('/home/hnu/Disk0/lmy/MAGC/imdb-A.pkl','rb')
    A = pickle.load(p)
    p1 = open('/home/hnu/Disk0/lmy/MAGC/imdb-B.pkl', 'rb')
    B = pickle.load(p1)
    p2 = open('/home/hnu/Disk0/lmy/MAGC/imdb-C.pkl', 'rb')
    C = pickle.load(p2)
    R = open('/home/hnu/Disk0/lmy/MAGC/imdb-L.pkl','rb')
    labels = pickle.load(R)
   # labels = labels[:100000]
   # adj = adj.tocsr()[:100000, :100000]
   # adj = adj.tocsr()
   # adj = adj.tay()
   # X = [X[:100000, :]]
    As = []
    Xs = []
    Xs.append(X)
    As.append(A)
    As.append(B)
    As.append(C)
    # X1 = X[:100000, :]
    # print(len(As))
    # print(len(Xs))
    # print("打印AS")
    # print(As)
    # print("打印Xs的shape")
    # print(X1.shape)
    #del adj, X
    return As, Xs, labels
def citeseer():
    dataset = '/home/hnu/Disk0/lmy/MAGC/citeseer'
    data = io.loadmat('{}.mat'.format(dataset))

    X = data['fea'].astype(float)
    A = data.get('W')
    labels = data['gnd']
    labels = labels.reshape(-1)

    As = [A]

    Xs = [X, np.log2(1+X)]

    return As, Xs, labels
def cora():
    dataset = '/home/hnu/Disk0/lmy/MAGC/cora'
    data = io.loadmat('{}.mat'.format(dataset))

    X = data['fea'].astype(float)
    A = data.get('W')
    labels = data['gnd']
    labels = labels.reshape(-1)

    As = [A]

    Xs = [X, np.log2(1+X)]

    return As, Xs, labels
def com():
    dataset = '/home/hnu/Disk0/lmy/MAGC/amzcomp'
    data = io.loadmat('{}.mat'.format(dataset))

    X = data['fea'].toarray().astype(float)
    A = data.get('W').toarray()
    labels = data['gnd']
    labels = labels.reshape(-1)
    #    X.toarray()
    #    A.toarray()
    As = [A]
    Xs = [X, X @ X.T]
   # Xs = [X, np.log2(1 + X)]

    return As, Xs, labels

def amap():
    #dataset = 'BlogCatalog'
    #data = io.loadmat('{}.mat'.format(dataset))

    #X = io.loadmat('amap-X.mat')#.toarray().astype(float)#data['Attributes'].toarray().astype(float)
    #A = io.loadmat('amap-A.mat').toarray()#data['Network'].toarray()  # .get('W')
    # X=np.vstack([X, X])
    # X=np.repeat(X,2, axis=0)
    #labels = io.loadmat('amap-labels.mat')#data['Label']
    f = open('/home/hnu/Disk0/lmy/MAGC/amap-A.pkl', 'rb')
    adj = pickle.load(f)
    p = open('/home/hnu/Disk0/lmy/MAGC/amap-X.pkl', 'rb')
    X = pickle.load(p)
    R = open('/home/hnu/Disk0/lmy/MAGC/amap-labels.pkl', 'rb')
    labels = pickle.load(R)
    labels = labels.reshape(-1)
    # labels = np.repeat(labels, 2, axis=0)
    # labels = np.hstack([labels,labels])
    # A = np.repeat(A, 2, axis=0)
    # A = np.repeat(A, 2, axis=1)
    As = [adj]

    Xs = [X, X @ X.T]

    return As, Xs, labels
def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
def datagen(dataset):
  if dataset == 'imdb': return imdb()
  if dataset == 'dblp': return dblp()
  if dataset == 'acm': return acm()
  if dataset == 'Amazon_photos': return photos()
  if dataset == 'wiki': return wiki()
  if dataset == 'Flickr': return Flickr()
  if dataset == 'cora': return cora()
  if dataset == 'arxiv': return arxiv()
  if dataset == 'BlogCatalog': return BlogCatalog()
  if dataset == 'com': return com()
  if dataset == 'amap': return amap()
  if dataset == 'citeseer': return citeseer()
  if dataset == 'Aminer': return Aminer()
  if dataset == 'imdb1': return IMDB1()


def preprocess_dataset(adj, features, tf_idf=False, beta=1):
  adj = adj + beta * sp.eye(adj.shape[0])
  rowsum = np.array(adj.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sp.diags(r_inv)
  adj = r_mat_inv.dot(adj)

  # if tf_idf:
  #      features = TfidfTransformer(norm='l2').fit_transform(features)
  # else:
  #      features = normalize(features, norm='l2')

  return adj, features


def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat
def calc_loss(x, x_aug, temperature=0.2, sym=True):
    batch_size = x.shape[0]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if sym:

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    #    print(pos_sim,sim_matrix.sum(dim=0))
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

    return loss

def cmat_to_psuedo_y_true_and_y_pred(cmat):
        y_true = []
        y_pred = []
        for true_class, row in enumerate(cmat):
            for pred_class, elm in enumerate(row):
                y_true.extend([true_class] * elm)
                y_pred.extend([pred_class] * elm)
        return y_true, y_pred

def clustering_accuracy(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


def clustering_f1_score(y_true, y_pred, **kwargs):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    pseudo_y_true, pseudo_y_pred = cmat_to_psuedo_y_true_and_y_pred(conf_mat)
    return metrics.f1_score(pseudo_y_true, pseudo_y_pred, **kwargs)

