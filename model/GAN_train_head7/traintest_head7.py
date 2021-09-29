import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import itertools
import struct
import argparse
from GAN import Generator
from GAN import Discriminator
import seaborn as sns
import time
import sys
import shutil
import jpholiday
import pandas as pd
import numpy as np
import scipy.sparse as ss
from torchsummary import summary

def show_loss_hist(hist, path):
    x = range(len(hist['D_losses_train']))
    y1 = hist['D_losses_train']
    y2 = hist['G_losses_train']
    y3 = hist['D_losses_test']
    y4 = hist['G_losses_test']
    plt.plot(x, y1, label='D_loss_train')
    plt.plot(x, y2, label='G_loss_train')
    plt.plot(x, y3, label='D_loss_test')
    plt.plot(x, y4, label='G_loss_test')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def show_RMSE_hist(hist, path):
    x = range(len(hist['RMSE_test']))
    y1 = hist['RMSE_test']
    plt.plot(x, y1, label='RMSE_test')
    plt.xlabel('Iter')
    plt.ylabel('RMSE')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# min-max normalization: x -> [-1,1]
def min_max_normal(x):
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)
    x = 2.0*x - 1.0
    return x # return x, min_x, max_x if we want to revert transform

def get_seq_data(data):
    seq_data = [data[i:i+opt.seq_len, ...] for i in range(0, data.shape[0]-opt.seq_len+1)]
    return np.array(seq_data)
        
def get_data(x, y, adj):
    # x = x.reshape(-1, seq_len, num_variable, 1)
    # y = y.reshape(-1, seq_len, num_variable, 4)
    # adj = adj.reshape(-1, seq_len, num_variable, num_variable)    
    x = x[:, :, np.newaxis].repeat(opt.channel, axis=2) # final_feat=channel=1
    x = min_max_normal(x)
    y = y[:, np.newaxis, ].repeat(opt.num_variable, axis=1)     # y is condition
    adj = adj[np.newaxis, :, :].repeat(x.shape[0], axis=0)
    return x, y, adj

def get_flow(): 
    inflow = np.load('../data/inflow_hour20180101_20210228.npy')
    outflow = np.load('../data/outflow_hour20180101_20210228.npy')
    odflow = ss.load_npz('../data/od_hour20180101_20210228.npz')
    odflow = np.array(odflow.todense()).reshape((-1, 47, 47))
    inflow, outflow, odflow = inflow[START_INDEX:END_INDEX+1, :], outflow[START_INDEX:END_INDEX+1, :], odflow[START_INDEX:END_INDEX+1, :, :]
    inflow, outflow, odflow = inflow[:, TARGET_AREA_ID], outflow[:, TARGET_AREA_ID], odflow[:, TARGET_AREA_ID, :][:, :, TARGET_AREA_ID]
    return inflow, outflow, odflow

def get_twitter():
    twitter = pd.read_csv('../data/Hagibis_twitter_count_by_prefecture.csv')
    twitter.columns = ['time', '神奈川県', '栃木県', '千葉県', '東京都', '埼玉県', '茨城県', '群馬県']
    twitter = twitter[['time'] + TARGET_AREA]
    timestamps = [date.strftime('%Y-%m-%d %H:%M:%S') for date in pd.date_range(START_DATE, END_DATE, freq='1H')]
    twitter = pd.merge(twitter, pd.DataFrame({'time': timestamps}), how='right')
    twitter = twitter.fillna(0)
    twitter = twitter[TARGET_AREA]
    return twitter.values

def get_onehottime():
    df = pd.DataFrame({'time': pd.date_range(START_DATE, END_DATE, freq='1H')})
    df['dayofweek'] = df.time.dt.weekday
    df['hourofday'] = df.time.dt.hour
    df['isholiday'] = df.apply(lambda x: int(jpholiday.is_holiday(x.time) | (x.dayofweek==5) | (x.dayofweek==6)), axis=1)
    tmp1 = pd.get_dummies(df.dayofweek)
    tmp2 = pd.get_dummies(df.hourofday)
    tmp3 = df[['isholiday']]
    df_dummy = pd.concat([tmp1, tmp2, tmp3], axis=1)
    return df_dummy.values

def get_adj():
    adj = np.load('../data/adjacency_matrix.npy')
    np.fill_diagonal(adj, 1)
    adj = adj[TARGET_AREA_ID, :][:, TARGET_AREA_ID]
    return adj

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = ss.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = ss.diags(d_inv_sqrt)
    return np.array(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense())

def traintest(D, G, x, y, adj, device):
    num_train_sample = int(x.shape[0] * opt.trainval_ratio)
    train_x, train_y, train_adj = x[:num_train_sample, ...], y[:num_train_sample, ...], adj[:num_train_sample, ...]
    test_x, test_y, test_adj = x[num_train_sample:, ...], y[num_train_sample:, ...], adj[num_train_sample:, ...]

    # dataset = Data.TensorDataset(x, y, adj)
    # train_loader = Data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)
    train_x, train_y, train_adj = torch.tensor(train_x).to(device), torch.tensor(train_y).to(device), torch.tensor(train_adj).to(device)
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y, train_adj)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)

    test_x, test_y, test_adj = torch.tensor(test_x).to(device), torch.tensor(test_y).to(device), torch.tensor(test_adj).to(device)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_adj)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size)  # no shuffle here.

    opt_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.l2)
    opt_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.l2)

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()

    loss_hist = {}
    loss_hist['D_losses_train'] = []
    loss_hist['G_losses_train'] = []
    loss_hist['D_losses_test'] = []
    loss_hist['G_losses_test'] = []
    loss_hist['RMSE_test'] = []

    for epoch in range(opt.epoch):
        # learning rate decay
        if epoch == 10:  # or epoch == 15:
            opt_G.param_groups[0]['lr'] /= 10
            opt_D.param_groups[0]['lr'] /= 10

        # training
        D_losses_train = []
        G_losses_train = []
        D.train()
        G.train()
        for step, (b_x, b_y, b_adj) in enumerate(train_loader):
            ######################### Train Discriminator #######################
            opt_D.zero_grad()
            num_seq = b_x.size(0)  # batch size of sequences
            real_seq = Variable(b_x.to(device)).float()  # put tensor in Variable
            seq_label = Variable(b_y.to(device)).float()
            seq_adj = Variable(b_adj.to(device)).float()
            prob_real_seq_right_pair = D(real_seq, seq_adj, seq_label)

            noise = torch.randn(num_seq, opt.seq_len * opt.init_dim * opt.num_variable).view(num_seq, opt.seq_len, opt.num_variable, opt.init_dim)
            noise = Variable(noise.to(device))  # randomly generate noise

            fake_seq = G(noise, seq_adj, seq_label)
            prob_fake_seq_pair = D(fake_seq, seq_adj, seq_label)

            # sample real seqs from database(just shuffle this batch seqs)
            shuffled_row_idx = torch.randperm(num_seq)
            real_shuffled_seq = b_x[shuffled_row_idx]
            real_shuffled_seq = Variable(real_shuffled_seq.to(device)).float()
            shuffled_adj = b_adj[shuffled_row_idx]
            shuffled_adj = Variable(shuffled_adj.to(device)).float()
            prob_real_seq_wrong_pair = D(real_shuffled_seq, shuffled_adj, seq_label)

            D_loss = - torch.mean(torch.log(prob_real_seq_right_pair) + torch.log(1. - prob_fake_seq_pair) + torch.log(1. - prob_real_seq_wrong_pair))
            D_loss.backward()
            opt_D.step()
            D_losses_train.append(D_loss.item())

            ########################### Train Generator #############################
            opt_G.zero_grad()
            noise2 = torch.randn(num_seq, opt.seq_len * opt.init_dim * opt.num_variable).view(num_seq, opt.seq_len,opt.num_variable, opt.init_dim)
            noise2 = Variable(noise2.to(device))

            # create random label
            y_real = Variable(torch.ones(num_seq).to(device))
            G_result = G(noise2, seq_adj, seq_label)
            D_result = D(G_result, seq_adj, seq_label).squeeze()
            G_loss = BCE_loss(D_result, y_real)
            G_loss.backward()
            opt_G.step()
            G_losses_train.append(G_loss.item())

        D_losses_train = torch.mean(torch.FloatTensor(D_losses_train)).item()
        G_losses_train = torch.mean(torch.FloatTensor(G_losses_train)).item()
        loss_hist['D_losses_train'].append(D_losses_train)
        loss_hist['G_losses_train'].append(G_losses_train)

        # testing
        rmse_test = []
        D_losses_test = []
        G_losses_test = []
        D.eval()
        G.eval()
        with torch.no_grad():
            for step, (b_x, b_y, b_adj) in enumerate(test_loader):
                ######################### Train Discriminator #######################
                num_seq = b_x.size(0)  # batch size of sequences
                real_seq = Variable(b_x.to(device)).float()  # put tensor in Variable
                seq_label = Variable(b_y.to(device)).float()
                seq_adj = Variable(b_adj.to(device)).float()
                prob_real_seq_right_pair = D(real_seq, seq_adj, seq_label)

                noise = torch.randn(num_seq, opt.seq_len * opt.init_dim * opt.num_variable).view(num_seq, opt.seq_len, opt.num_variable, opt.init_dim)
                noise = Variable(noise.to(device))  # randomly generate noise

                fake_seq = G(noise, seq_adj, seq_label)
                prob_fake_seq_pair = D(fake_seq, seq_adj, seq_label)

                # sample real seqs from database(just shuffle this batch seqs)
                shuffled_row_idx = torch.randperm(num_seq)
                real_shuffled_seq = b_x[shuffled_row_idx]
                real_shuffled_seq = Variable(real_shuffled_seq.to(device)).float()
                shuffled_adj = b_adj[shuffled_row_idx]
                shuffled_adj = Variable(shuffled_adj.to(device)).float()
                prob_real_seq_wrong_pair = D(real_shuffled_seq, shuffled_adj, seq_label)

                D_loss = - torch.mean(torch.log(prob_real_seq_right_pair) + torch.log(1. - prob_fake_seq_pair) + torch.log(1. - prob_real_seq_wrong_pair))
                D_losses_test.append(D_loss.item())

                ########################### Train Generator #############################
                opt_G.zero_grad()
                noise2 = torch.randn(num_seq, opt.seq_len * opt.init_dim * opt.num_variable).view(num_seq, opt.seq_len, opt.num_variable, opt.init_dim)
                noise2 = Variable(noise2.to(device))

                # create random label
                y_real = Variable(torch.ones(num_seq).to(device))
                G_result = G(noise2, seq_adj, seq_label)
                D_result = D(G_result, seq_adj, seq_label).squeeze()
                G_loss = BCE_loss(D_result, y_real)
                G_losses_test.append(G_loss.item())

                pred = fake_seq.cpu().data.numpy()
                truth = b_x.cpu().data.numpy()
                rmse_test.append(np.sqrt(np.mean(np.square(pred - truth))))

        D_losses_test = torch.mean(torch.FloatTensor(D_losses_test)).item()
        G_losses_test = torch.mean(torch.FloatTensor(G_losses_test)).item()
        rmse_test = np.mean(rmse_test)
        loss_hist['D_losses_test'].append(D_losses_test)
        loss_hist['G_losses_test'].append(G_losses_test)
        loss_hist['RMSE_test'].append(rmse_test)

        print('Epoch', epoch, time.ctime(), 'D_loss_train, G_loss_train, D_loss_test, G_loss_test', D_losses_train, G_losses_train, D_losses_test, G_losses_test)

        if (epoch + 1) % 100 == 0:
            fake = fake_seq.cpu().data.numpy()
            truth = b_x.cpu().data.numpy()
            print(fake.shape, truth.shape) # the last batch size 839%64=23
            sns.set_style('darkgrid')
            fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 5))
            ax[0].plot(fake[0, :, :, 0])
            ax[0].legend(TARGET_AREA_EN)
            ax[0].set_title('Generated Time Series')
            ax[0].set_ylim(-1.0, 1.0)
            ax[1].plot(truth[0, :, :, 0])
            ax[1].legend(TARGET_AREA_EN)
            ax[1].set_title('Ground-truth Time Series')
            ax[1].set_ylim(-1.0, 1.0)
            plt.savefig('{}/fake_seqs_{}.png'.format(path, epoch + 1))
            plt.close()

    show_loss_hist(loss_hist, path=f'{path}/loss_hist.png')
    show_RMSE_hist(loss_hist, path=f'{path}/RMSE_hist.png')
    torch.save(G.state_dict(), f'{path}/G_params.pkl')  # save parameters
    torch.save(D.state_dict(), f'{path}/D_params.pkl')

DAYS = [date.strftime('%Y-%m-%d %H:%M:%S') for date in pd.date_range(start='2018-01-01 00:00:00', end='2021-02-28 23:59:59', freq='1H')]
START_DATE, END_DATE = '2018-10-27 09:00:00', '2019-10-27 08:00:00'
START_INDEX, END_INDEX = DAYS.index(START_DATE), DAYS.index(END_DATE)
TARGET_AREA = ['茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都', '神奈川県'] # should correspond to area id.
TARGET_AREA_EN = ['Ibaraki', 'Tochigi', 'Gunma', 'Saitama', 'Chiba', 'Tokyo', 'Kanagawa']
TARGET_AREA_ID = [7, 8, 9, 10, 11, 12, 13] # should correspond to area id.

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000, help="number of epochs of training") # original 1500
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--adj_bar", type=float, default=0.47, help="adj bar")
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--l2', type=float, default=0.1, help='l2 penalty')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--seq_len', type=int, default=12, help='sequence length of images, which should be even nums (2,4,6,12)')
parser.add_argument('--num_head', type=int, default=7, help='number of heads in self-attention')
#parser.add_argument('--num_block', type=int, default=2, help='repeating times of buiding block') # original 3
parser.add_argument('--num_block_D', type=int, default=2, help='repeating times of buiding block for D') # original 3
parser.add_argument('--num_block_G', type=int, default=3, help='repeating times of buiding block for G') # original 3
parser.add_argument('--num_variable', type=int, default=7, help='total number of the target variables') # current 7
parser.add_argument('--D_hidden_feat', type=int, default=16, help='hidden features of D')
parser.add_argument('--G_hidden_feat', type=int, default=64, help='hidden features of G')
parser.add_argument('--channel', type=int, default=1, help='channel')
parser.add_argument('--D_final_feat', type=int, default=1, help='output features of D')
parser.add_argument('--G_final_feat', type=int, default=1, help='output features of G')
parser.add_argument("--init_dim", type=int, default=100, help="dimensionality of the latent code")
parser.add_argument('--D_input_feat', type=int, default=1+32, help='input features of D (include init features and num of conditions')
parser.add_argument('--G_input_feat', type=int, default=100+32, help='input features of G (include init features and num of conditions')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the total ratio of training data and validation data')
parser.add_argument('--val_ratio', type=float, default=0.2, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--which_gpu', type=int, default=3, help='which gpu to use')
opt = parser.parse_args()

#path = './GAN_train' + '_' + time.strftime('%Y%m%d%H%M', time.localtime())
path = 'GAN_train_head7'

def main():
    if not os.path.exists(path):
        os.mkdir(path)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, path)
    shutil.copy2('GAN.py', path)
    
    device = torch.device("cuda:{}".format(opt.which_gpu)) if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    inflow, outflow, odflow = get_flow()
    twitter = get_twitter()
    onehottime = get_onehottime()
    adj01 = get_adj()
    adj = sym_adj(adj01)
    print(inflow.shape, outflow.shape, odflow.shape, twitter.shape, onehottime.shape, adj.shape)
    
    x, y, adj = get_data(inflow, onehottime, adj)       # inflow channel only
    seq_x, seq_y, seq_adj = get_seq_data(x), get_seq_data(y), get_seq_data(adj)
    print(seq_x.shape, seq_y.shape, seq_adj.shape, seq_x.min(), seq_x.max(), seq_adj.min(), seq_adj.max())
    # num_train_sample = int(seq_x.shape[0] * opt.trainval_ratio)
    # train_seq_x, train_seq_y, train_seq_adj = seq_x[:num_train_sample, ...], seq_y[:num_train_sample, ...], seq_adj[:num_train_sample, ...]
    # test_seq_x, test_seq_y, test_seq_adj = seq_x[num_train_sample:, ...], seq_y[num_train_sample:, ...], seq_adj[num_train_sample:, ...]
    # print(train_seq_x.shape, train_seq_y.shape, train_seq_adj.shape)
    # print(test_seq_x.shape, test_seq_y.shape, test_seq_adj.shape)
    
    D = Discriminator(opt.D_input_feat, opt.D_hidden_feat, opt.D_final_feat, opt.num_head, opt.dropout, opt.num_block_D, opt.num_variable, opt.seq_len).to(device)
    G = Generator(opt.G_input_feat, opt.G_hidden_feat, opt.G_final_feat, opt.num_head, opt.dropout, opt.num_block_G, opt.num_variable, opt.seq_len).to(device)
    
    start = time.ctime()
    traintest(D, G, seq_x, seq_y, seq_adj, device)
    end = time.ctime()
    print('start and end time for total training process...', start, end)
    
if __name__ == '__main__':
    main()
