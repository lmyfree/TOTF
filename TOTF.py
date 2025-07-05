import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from get_mask import generate_uniform_mask, z_score_normalize, mean1
from evaluation import eva
from utils import clustering_accuracy, clustering_f1_score, preprocess_dataset, datagen
from utils1 import load_data, compute_ppr, get_sharp_common_z, sample_graph, normalize_weight
from torch.optim import Adam
from itertools import product
from impute2_bei import imputation,sample_M
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from models_cora_new import commonExtractor
# ============================ 1.parameters ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='acm', help='acm, dblp, imdb, cora, citeseer, pubmed')
parser.add_argument('--train', type=bool, default=False, help='training mode')
parser.add_argument('--model_name', type=str, default='samgc_acm', help='model name')

parser.add_argument('--path', type=str, default='/home/hnu/Disk0/lmy/SAMGC-main/data/', help='')
parser.add_argument('--order', type=int, default=16, help='aggregation orders')  # cora=[8,6] citeseer=4 acm=[16] dblp=9
parser.add_argument('--weight_soft', type=float, default=0.5, help='parameter of p')  # acm=0, dblp=[0.2,0.3]
parser.add_argument('--lam_emd', type=float, default=1., help='trade off between global self-attention and gnn')
parser.add_argument('--kl_step', type=float, default=10., help='lambda kl')

parser.add_argument('--lam_consis', type=float, default=10., help='lambda consis')
parser.add_argument('--hidden_dim', type=int, default=256, help='lambda consis')  # citeseer=[512] others=default 256
parser.add_argument('--latent_dim', type=int, default=64, help='lambda consis')  # citeseer=[16] others=default  64

parser.add_argument('--epoch', type=int, default=200, help='')
parser.add_argument('--patience', type=int, default=100, help='')
parser.add_argument('--lr', type=float, default=1e-4, help='')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='')
parser.add_argument('--temperature', type=float, default=0.5, help='')
parser.add_argument('--cuda_device', type=int, default=0, help='')
parser.add_argument('--use_cuda', type=bool, default=True, help='')
parser.add_argument('--update_interval', type=int, default=1, help='')
parser.add_argument('--random_seed', type=int, default=2022, help='')
parser.add_argument('--add_graph', type=bool, default=True, help='')


args = parser.parse_args()

train = args.train
dataset = args.dataset  # [imdb, dblp, acm]  [cora, citeseer, pubmed]
path = args.path
order = args.order  # acm=16, dblp=9,10, imdb=0,1,2
weight_soft = args.weight_soft # acm=0., dblp= [0.0-0.5], imdb
kl_step = args.kl_step  # acm=0.09,10. dblp = 1., imdb
kl_max = kl_step  # acm=10 dblp=100, imdb
lam_consis = args.lam_consis  # acm=10 dblp=1 current0.5, imdb
lam_emd = args.lam_emd

add_graph=args.add_graph
hidden_dim = args.hidden_dim
latent_dim = args.latent_dim
epoch = args.epoch
patience = args.patience
lr = args.lr
weight_decay = args.weight_decay
temprature = args.temperature
cuda_device = args.cuda_device
use_cuda = args.use_cuda
update_interval = args.update_interval
random_seed = args.random_seed

torch.manual_seed(random_seed)

# ============================ 2.dataset and model preparing ==========================

feat_dim = [0,0]
   # data = sio.loadmat('{}.mat'.format(dataset))
dataset = 'cora'
print('-----------------', dataset, '-----------------')
As, Xs, labels = datagen(dataset)
k = len(np.unique(labels))
views = list(product(As, Xs))
class_num = int(labels.max())
tM=[]
for v in range(len(views)):
     A, X = views[v]
     tf_idf = dataset in ['acm', 'dblp', 'imdb']
     norm_adj, features = preprocess_dataset(A, X, tf_idf=tf_idf, beta=1)
     trainM = sample_M(len(features), len(features[0, :]), 0.1)
     tM.append(trainM)
     features = features*trainM
     #features = normalize(features, norm='l2')
     features = TfidfTransformer(norm='l2').fit_transform(features)
#
     if type(features) != np.ndarray:
         features = features.toarray()
#
     if type(norm_adj) == np.matrix:
         norm_adj = np.asarray(norm_adj)
#
     views[v] = (norm_adj, features)
     adj, X = views[v]
     X = X.astype(np.float32)
     adj = adj.astype(np.float32)
     X = torch.tensor(X)
     norm_adj = torch.tensor(adj)
     f, X1 = imputation(X, norm_adj, v, trainM, feat_dim, hidden_dim, latent_dim, class_num, lam_emd=lam_emd,
                        order=order)
     norm_adj = torch.tensor(norm_adj)
   #  X = X1.detach()
     views[v] = (norm_adj, X)
     feat_dim[v] = X.size(1)
# for i in range(2):
#     # Normalize A
#     adj, X = views[i]
#     X = X.astype(np.float32)
#     adj = adj.astype(np.float32)
#     X = torch.tensor(X)
#     norm_adj = torch.tensor(adj)
#   #  views[i] = (norm_adj, X)
#     trainM1=tM[i]
#     f, X1 = imputation(X, norm_adj,i,trainM1,feat_dim, hidden_dim, latent_dim, class_num, lam_emd=lam_emd, order=order)
#    # X = f.detach()
#     #  adj= adj.toarray()
#     views[i] = (norm_adj, X)
#     feat_dim[i] = X.size(1)


# graph_num = 1
# adjs=adjs[:1]
# adjs_labels = adjs_labels[:1]
#for i in np.arange(0, 1).reshape(-1):
#    features = z_score_normalize(features)

#feat_dim = features.shape[1]
#Data = features.numpy()
train_rate = 1
p_miss = 0.4
graph_num = len(views)
No = len(Xs[0])
adjs = norm_adj
#p_miss_vec = p_miss * np.ones((Dim, 1))
#print(features)
#Missing = np.zeros((No, Dim))
#for i in range(Dim):
#        A = np.random.uniform(0., 1., size=[len(Data), ])
#        B = A > p_miss_vec[i]
#        Missing[:, i] = 1. * B
#mask = generate_uniform_mask(features, 0)
#print(features.type)
#features = features * (mask == False) + (0) * (mask == True)
#np.transpose(mean1(features)) * (mask == True)
#Train_No = int(No * train_rate)
#idx = np.random.permutation(No)
#trainX = Data#[idx[:Train_No], :]
#trainM = Missing#[idx[:Train_No], :]
#print(trainM)
#trainM=trainM * trainX
#features = torch.from_numpy(trainX).float()
        # mask the data
print("111111")
#print(features.shape)
#print(mask)
#features = features * mask[:, 0][:, np.newaxis]
        #x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]


    # drop_adj, drop_adj_labels = sample_graph(adj_labels*1.0, drop_rate=drop_rate)

        # adjs_labels.append(ppr_adj_labels)
    # adj_labels = ppr_adj_labels



model = commonExtractor(feat_dim, hidden_dim, latent_dim, class_num, lam_emd=lam_emd, order=order, view_num=graph_num)

#model = MultiGraphAutoEncoder(feat_dim, hidden_dim, latent_dim, class_num, lam_emd=lam_emd, order=order, view_num=graph_num)

# if use_cuda:
#     torch.cuda.set_device(cuda_device)
#     torch.cuda.manual_seed(random_seed)
#     model = model.cuda()
#     adjs = adj.cuda()#[a.cuda() for a in adj]
#    # adjs_labels = [adj_labels.cuda() for adj_labels in adjs_labels]
#     features = [torch.tensor(b.astype(np.float32)).cuda() for b in Xs]#.cuda()
#
#
# device = features[0].device

# ------------------------------------------- optimizer -------------------------------
# optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
param_ge = []
param_ae = []
model_dict = model.state_dict()
pre_enc0=torch.load('view0_enc.pth')
# #pre_dec0=torch.load('view0_dec.pth')
# #for name,param in model.named_parameters():
# #	print(name,param)
# #model_dict = model.state_dict()
model_dict["graphenc_0.gnn_1.weight"] = pre_enc0["gnn_1.weight"]#torch.ones((1, 1, 3, 3, 3))
model_dict["graphenc_0.gnn_2.weight"] = pre_enc0["gnn_2.weight"]
model_dict["graphenc_0.gnn_3.weight"] = pre_enc0["gnn_3.weight"]
#
pre_enc1=torch.load('view1_enc.pth')
#
# #for name,param in model.named_parameters():
# #	print(name,param)
# #model_dict = model.state_dict()
model_dict["graphenc_1.gnn_1.weight"] = pre_enc1["gnn_1.weight"]#torch.ones((1, 1, 3, 3, 3))
model_dict["graphenc_1.gnn_2.weight"] = pre_enc1["gnn_2.weight"]
model_dict["graphenc_1.gnn_3.weight"] = pre_enc1["gnn_3.weight"]
#
torch.save(model_dict, 'model_0_.pth')
# # 验证修改是否成功
model.load_state_dict(torch.load('model_0_.pth'))
for i in range(graph_num):
   # param_ge.append({'params': model.GraphEnc[i].parameters()})
    #param_ae.append({'params': model.FeatDec[i].parameters()})

    param_ae.append({'params': model.cluster_layer[i]})
param_ae.append({'params': model.cluster_layer[graph_num]})
param_ae.append({'params': model.LatentMap.parameters()})

optimizer_ge = Adam(param_ge + param_ae,
                 lr=lr, weight_decay=weight_decay)

# cluster parameter initiate
#y = labels.cpu().numpy()
y = labels-1


# ============================ 3.Training ==========================
if train:
    with torch.no_grad():
        zs = []
        qs = []
        kmeans = KMeans(n_clusters=class_num, n_init=3)
        zs, qs, common_z = model(views, zs, qs, graph_num)
        for z in zs:
            y_pred = kmeans.fit_predict(z.data.cpu().numpy())
            y_pred_last = y_pred
            model.cluster_layer[i].data = torch.tensor(kmeans.cluster_centers_)#.to(device)
            eva(y, y_pred, 'K{}'.format(i))
        #tz = torch.cat([common_z for i in range(graph_num)], dim=-1)
       # tz = torch.cat(zs, dim=-1)
        tz = torch.cat([common_z for i in range(graph_num)], dim=-1)
        y_pred = kmeans.fit_predict(tz.data.cpu().numpy())
        y_pred_last = y_pred
        model.cluster_layer[-1].data = torch.tensor(kmeans.cluster_centers_)  # .to(device)
        eva(y, y_pred, 'Kz')
       # print()

        #print()
   # for iv in range(graph_num):
   #         features, adjs[iv]= imputation(features, adjs[iv])
    bad_count = 0
    best_loss = 100
    best_acc = 1e-12
    best_nmi = 1e-12
    best_ari = 1e-12
    best_f1 = 1e-12
    best_epoch = 0
    l = 0.0
    #print(features.shape)
    best_a = [1e-12 for i in range(graph_num)]
    weights = normalize_weight(best_a)

    # for i in range(num_graph+1):
    #     model.cluster_layer[i].requires_grad = False

    for epoch_num in range(epoch):
        # drop_adj, drop_adj_labels = sample_graph(adj_labels.clone(), drop_rate=drop_rate)
        # adjs[1] = drop_adj.to(device)
        # adjs_labels[1] = drop_adj_labels.to(device)
        model.train()
        print(epoch_num)
        zs = []
        x_preds = []
        qs = []
        re_loss = 0.
        consis_loss = 0.
        re_feat_loss = 0.
        kl_loss = 0.
        kl_loss1 = 0.
        kl_loss2 = 0.
        zs, qs, common_z = model(views, zs, qs, graph_num)





        # ------------------------------------- consistency -----------------------------------
        for z in zs:
            # consis_loss += F.mse_loss(zs[0], zs[1])
            consis_loss += F.mse_loss(common_z, z)
        # consis_loss /= graph_num
        consis_loss *= lam_consis

        # ---------------------------------------- kl loss------------------------------------
        h = torch.cat([common_z for i in range(graph_num)], dim=-1)#torch.cat([zs[i] * weights[i] for i in range(graph_num)], dim=-1)

        qh = model.predict_distribution(h, -1)
        p = model.target_distribution(qh)
        kl_loss += F.kl_div(qh.log(), p, reduction='batchmean')
        for i in range(graph_num):
            kl_loss += F.kl_div(qs[i].log(), p, reduction='batchmean')
            #kl_loss += F.kl_div(qs[i].log(), model.target_distribution(qs[i]), reduction='batchmean')
            # kl_loss += F.kl_div(qs[i].log(), model.target_distribution(qs[i]), reduction='batchmean')

        if l < kl_max:
            l = kl_step * (epoch_num+1)
        else:
            l = kl_max
        kl_loss *= l
        # -----------------------------------------------------------------------

        #loss = re_loss + kl_loss + consis_loss + re_feat_loss

        loss = consis_loss+ kl_loss
       # z=z.detach()#.requires_grad = False
        print("loss",loss)
        optimizer_ge.zero_grad()
        loss.backward()
        optimizer_ge.step()

    # ============================ 4.evaluation ==========================
        if epoch_num % update_interval == 0:  # [1,3,5]
            model.eval()
            with torch.no_grad():
                # update_interval
                zs = []
                qs = []
                q = 0.
                zs, qs, common_z = model(views, zs, qs, graph_num)
            #tz = torch.cat(zs, dim=-1)
            tz = torch.cat([common_z for i in range(graph_num)], dim=-1)
            q = model.predict_distribution(tz, -1)
            kmeans = KMeans(n_clusters=class_num, n_init=20)
            res2 = kmeans.fit_predict(common_z.data.cpu().numpy())
            nmi, acc, ari, f1 = eva(y, res2, str(epoch_num) + 'Kz')

            # for i in range(graph_num):
            #     res1 = kmeans.fit_predict(zs[i].data.cpu().numpy())
            #     _, _, _, _ = eva(y, res1, str(epoch_num) + 'K'+str(i))

         #   for i in range(graph_num):
         #       print('view:', str(i), np.around(model.GraphEnc[i].la.data.cpu().numpy(), 3))
         #   print(weights)
            model.train()
    # ======================================= 5. postprocess ======================================
        print(#'Epoch:{}'.format(epoch_num),
              'bad_count:{}'.format(bad_count),
              'kl:{:.4f}'.format(kl_loss),
              'consis:{:4f}'.format(consis_loss),
            #  'rec:{:.4f}'.format(re_loss.item()),
             # 're_feat:{:.4f}'.format(re_feat_loss.item()),
              end='\n')

        if acc > best_acc:
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
            best_epoch = epoch_num
            if loss < best_loss:
                best_loss = loss
            print('saving model epcoh:{}'.format(epoch_num))
            torch.save({'state_dict':model.state_dict(),
                        'weights': weights}, 'samgc_{}.pkl'.format(dataset))
            bad_count = 0
        else:
            bad_count += 1

        print('best acc:{}, best nmi:{}, best ari:{}, best f1:{},best loss:{}, bestepoch:{}'.format(
            best_acc, best_nmi, best_ari, best_f1, best_loss, best_epoch))
        print()

        if bad_count >= patience:
            print('complete training, best acc:{}, best nmi:{}, best ari:{}, best f1:{},best loss:{}, bestepoch:{}'.format(
                best_acc, best_nmi, best_ari, best_f1, best_loss, best_epoch))
            break


# ============================================== Test =====================================================
if not train:
    model_name = args.model_name
else:
    model_name = 'samgc_{}.pkl'.format(dataset)
print('Loading model:{}...'.format(model_name))
#bestmodel = torch.load(model_name, map_location=features[0].device)
best_model = torch.load(model_name)
weights = best_model['weights']
print(weights)
state_dict = best_model['state_dict']
model.load_state_dict(state_dict)
print('Evaluating....')
print('Evaluating....')
for epoch_num in range(30):
     with torch.no_grad():
        # update_interval
        zs = []
        qs = []
        q = 0.
        zs, qs, common_z = model(views, zs, qs, graph_num)

     #z = get_sharp_common_z(zs, temp=temprature)
   #  tz = torch.cat([common_z for i in range(graph_num)], dim=-1)
     #z = torch.cat([zs[i] for i in range(graph_num)], dim=-1)
     #tz = torch.cat(zs, dim=-1)
     kmeans = KMeans(n_clusters=class_num, n_init=20)
     #res2 = kmeans.fit_predict(common_z.data.cpu().numpy())
     res2 = kmeans.fit_predict(common_z.data.cpu().numpy())
     nmi, acc, ari, f1 = eva(y, res2, str('eva:') + 'Kz')
     print('Results: acc:{},  nmi:{},  ari:{},  f1:{}, '.format(
         acc, nmi, ari, f1))

