import torch
import numpy as np
import os
# from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from IGAE1 import IGAE_encoder, IGAE_decoder
import torch.nn.functional as F
from functools import partial
from utils import sce_loss
from torch.nn.parameter import Parameter
import torch.nn as nn

from layers import GraphEncoder, LatentMappingLayer

# Mask Vector and Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
    return C


# 2. Plot (4 x 4 subfigures)
def plot(samples):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


# %% 3. Others
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 1., size=[m, n])


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


class NetD(torch.nn.Module):
    def __init__(self,Dim):
        super(NetD, self).__init__()
        self.fc1 = torch.nn.Linear(Dim * 2, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, Dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

    def forward(self, x, m, g, h):
        """Eq(3)"""
        inp = h * x + (1 - h) * g
        inp = torch.cat((inp, g), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        #         out = self.sigmoid(self.fc3(out)) # [0,1] Probability Output
        out = self.fc3(out)

        return out


"""
Eq(2)
"""
class NetG1(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, class_num, lam_emd=1., alpha=0.2, order=5):
        super(NetG1, self).__init__()
        self.hidden_size = hidden_dim
        self.latent_dim = latent_dim
        #feat_dim = 1433
       # hidden_dim = 256
       # latent_dim = 64
        self.GraphEnc = IGAE_encoder(gae_n_enc_1=128, gae_n_enc_2=256, gae_n_enc_3=64, n_input=feat_dim)  # GraphEncoder(feat_dim, hidden_dim, lam_emd=lam_emd, order=order)

        self.LatentMap = LatentMappingLayer(64, hidden_dim, 64)
        self.FeatDec = IGAE_decoder(gae_n_dec_1=64, gae_n_dec_2=256, gae_n_dec_3=128,n_input=feat_dim)
    def forward(self, x, m,adj):
        e = self.GraphEnc(x, adj)
        # z = torch.tensor(z)
        # print('view:', view, self.GraphEnc[view].la)
        # print(self.cluster_layer[view])
        z = self.LatentMap(e)
      #  z = m*z
        z_norm = F.normalize(z, p=2, dim=1)
        A_pred = self.decode(z_norm)
        # q = self.predict_distribution(z_norm, view)

        x_prim = self.FeatDec(z, adj)
        # x_pred = torch.sigmoid(x_prim)
        print("over")
        return x_prim, A_pred
        # [0,1] Probability Output
            #         out = self.fc3(out)


    def decode(self, z):
        rec_graph = torch.sigmoid(torch.matmul(z, z.T))
        return rec_graph

class NetG(torch.nn.Module):
    def __init__(self,Dim):
        super(NetG, self).__init__()
        self.fc1 = torch.nn.Linear(Dim * 2, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, Dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

    def forward(self, x, z, m):
        inp = m * x + (1 - m) * z
        inp = torch.cat((inp, m), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))  # [0,1] Probability Output
        #         out = self.fc3(out)

        return out


def imputation(features, adjs,i,trainM,feat_dim, hidden_dim, latent_dim, class_num, lam_emd, order):
    # 1. Mini batch size

    hidden_dim = 256
    latent_dim = 64
    # 2. Missing rate
    p_miss = 0.1
    # 3. Hint rate
    p_hint = 0.4
    # 4. Loss Hyperparameters
    alpha = 1
    # 5. Imput Dim (Fixed)

    # 6. No

    # dataset_file = '/home/hnu/Disk0/lmy/SAMGC-main/Spam.csv'
    Data = features.cpu().detach().numpy()
    trainX = Data  # np.loadtxt(dataset_file, delimiter=",", skiprows=1)
    testX = Data  # np.loadtxt(dataset_file, delimiter=",", skiprows=1)
    No = len(trainX)
    feat_dim = len(trainX[0, :])
    Dim = len(trainX[0, :])
    Train_No = No
    Test_No = No
    mb_size = No
    #trainM = sample_M(No, Dim, p_miss)
    testM = sample_M(No, Dim, p_miss)
    # t1=np.array([[1,2,3],[1,2,2]])
    # t2=np.array([[0,1,0],[1,0,1]])
    # t3=t1*t2
    # print("t1",t1)
    # print("t2", t2)
    # print("t3", t3)
    EX=trainX
    trainX = trainX*trainM
    X= trainX
    X = torch.tensor(X)
    netD = NetD(Dim)
    netG = NetG1(feat_dim, hidden_dim, latent_dim, class_num, lam_emd=lam_emd, order=order)
    X_mb = trainX
    optimD = torch.optim.Adam(netD.parameters(), lr=0.001)
    optimG = torch.optim.Adam(netG.parameters(), lr=0.001)

    # Output Initialization
    if not os.path.exists('Multiple_Impute_out1/'):
        os.makedirs('Multiple_Impute_out1/')

    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean")
    mse_loss = torch.nn.MSELoss(reduction="elementwise_mean")
    #mse_loss = partial(sce_loss, alpha=2)
    M_mb = trainM
    H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
    H_mb = M_mb * H_mb1 + 0.5 * (1 - H_mb1)

    # Missing Data Introduce
    trainX = torch.tensor(X_mb).float()
    X_mb = torch.tensor(X_mb).float()
    # New_X_mb = torch.tensor(New_X_mb).float()
    #Z_mb = torch.tensor(Z_mb).float()
    M_mb = torch.tensor(M_mb).float()
    H_mb = torch.tensor(H_mb).float()
    # X_mb=X_mb.numpy()
    # M_mb=M_mb.numpy()
    # H_mb=H_mb.numpy()
    #i = 1
    # %% Start Iterations
    for it in range(100):
        # %% Inputs
        #mb_idx = sample_idx(Train_No, mb_size)
          # [mb_idx,:]
        #Z_mb = sample_Z(mb_size, Dim)

       # M_mb = trainM  # [mb_idx,:]


        # Train D
        #print(torch.is_tensor(X_mb))
        G_sample, A = netG(X_mb, M_mb, adjs)
        D_prob = netD(X_mb, M_mb, G_sample, H_mb)
        D_loss = bce_loss(D_prob, M_mb)
        optimD.zero_grad()
        D_loss.backward()
        optimD.step()

        # Train G
        G_sample ,A = netG(X_mb, M_mb, adjs)#.detach()
        D_prob = netD(X_mb, M_mb, G_sample, H_mb)
       # D_prob.detach_()
        G_loss1 = ((1 - M_mb) * (torch.sigmoid(D_prob) + 1e-8).log()).mean() / (1 - M_mb).sum()
        G_mse_loss = mse_loss(M_mb * trainX, M_mb * G_sample)
        #G_mse_loss = mse_loss(M_mb * G_sample, M_mb * trainX)# / M_mb.sum()
        G_re_loss = F.binary_cross_entropy(A, adjs)
        G_loss = G_loss1 + 1 * G_mse_loss #+ 0.1 * G_re_loss
        optimG.zero_grad()
        G_loss.backward()
        optimG.step()

        #X_mb = (M_mb * trainX + (1 - M_mb) * G_sample).detach()
        G_mse_test = mse_loss(M_mb * X_mb, M_mb * G_sample) #/ (1 - M_mb).sum()
        if (it == 99):
            X_mb=torch.tensor(X_mb)
            print("M",M_mb.type)
            print("X",X_mb.type)
           # print("trainM", trainM.type)
            print("G_sample", G_sample.type)
            trainM=torch.tensor(trainM)
            X = M_mb * trainX +(1 - trainM) * G_sample#torch.tensor(EX)#M_mb * trainX+(1 - trainM) * G_sample
            X = X.float()
            if (i==0):
                torch.save(netG.GraphEnc.state_dict(), 'view0_enc.pth')
                torch.save(netG.FeatDec.state_dict(), 'view0_dec.pth')
            if (i==1):
                torch.save(netG.GraphEnc.state_dict(), 'view1_enc.pth')
                torch.save(netG.FeatDec.state_dict(), 'view1_dec.pth')
            if (i==2):
                torch.save(netG.GraphEnc.state_dict(), 'view2_enc.pth')
                torch.save(netG.FeatDec.state_dict(), 'view2_dec.pth')
            if (i==3):
                torch.save(netG.GraphEnc.state_dict(), 'view3_enc.pth')
                torch.save(netG.FeatDec.state_dict(), 'view3_dec.pth')
              #torch.save(netG, './netG.pth')
            #if (i == 1):
            #    torch.save(netG, './netG1.pth')
            #if (i == 2):
            #    torch.save(netG, './netG2.pth')
            #if (i == 3):
           # torch.save(netG.GraphEnc.state_dict(), 'phase1_enc.pth')
           # torch.save(netG.FeatDec.state_dict(), 'phase1_dec.pth')

            #torch.save(netG, './netG3.pth')

            #model_enc = netG.GraphEnc.state_dict()
            #model_dec = netG.FeatDec.state_dict()


        # %% Output figure
        # if it % 100 == 0:
        #
        #     mb_idx = sample_idx(Test_No, 5)
        #     X_mb = testX#[mb_idx,:]
        #     M_mb = testM#[mb_idx,:]
        #     Z_mb = sample_Z(Test_No, Dim)
        #
        #     New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
        #
        #     X_mb = torch.tensor(X_mb).float()
        #     New_X_mb = torch.tensor(New_X_mb).float()
        #     Z_mb = torch.tensor(Z_mb).float()
        #     M_mb = torch.tensor(M_mb).float()
        #
        #     samples1 = X_mb
        #     samples5 = M_mb * X_mb + (1-M_mb) * Z_mb
        #
        #     samples2 = netG(X_mb, New_X_mb, M_mb)
        #     samples2 = M_mb * X_mb + (1-M_mb) * samples2
        #
        #     Z_mb = torch.Tensor(sample_Z(5, Dim)).float()
        #     New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
        #
        #     samples3 =netG(X_mb, New_X_mb, M_mb)
        #     samples3 = M_mb * X_mb + (1-M_mb) * samples3
        #
        #     Z_mb = torch.tensor(sample_Z(5, Dim)).float()
        #     New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
        #     samples4 = netG(X_mb, New_X_mb, M_mb)
        #     samples4 = M_mb * X_mb + (1-M_mb) * samples4
        #
        #
        #     samples = np.vstack([samples5.detach().data, samples2.detach().data, samples3.detach().data,
        #                          samples4.detach().data, samples1.detach().data])
        #
        #     # fig = plot(samples)
        #     # plt.savefig('Multiple_Impute_out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        #     i += 1
        # plt.close(fig)

        # %% Intermediate Losses
        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('D_loss: {:.4}'.format(D_loss))
            print('Train_loss: {:.4}'.format(G_mse_loss))
            print('Test_loss: {:.4}'.format(G_mse_test))
            print()
    return M_mb * trainX, X#X#, model_enc, model_dec