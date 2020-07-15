import sys
sys.path.insert(0, '/gpfs/share/home/1601210097/projects/style_transfer_aaai/stylenas/')
import numpy as np
import torch
import os
import cv2
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data.sampler import RandomSampler
import torch.autograd as autograd
from tqdm import tqdm
from matplotlib import pyplot as plt
from dataset import TransferDataset
from torch.utils.serialization import load_lua
import argparse
from models_photorealistic_nas.VGG_with_decoder import encoder, decoder

abs_dir = os.path.abspath(os.path.dirname(__file__))

def load_nets():
    encoder_param = load_lua('/gpfs/share/home/1601210097/projects/style_transfer_aaai/stylenas/models_photorealistic_nas/vgg_normalised_conv5_1.t7')
    net_e = encoder(encoder_param)
    net_d = decoder()
    return net_e, net_d


def get_gram_matrix(f):
    n, c, h, w = f.size(0), f.size(1), f.size(2), f.size(3)
    f = f.view(n, c, -1)
    gram = f.bmm(f.transpose(1, 2)) / (h * w)
    return gram


def get_loss(encoder, decoder, content, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control):
    fc = encoder(content)
    content_new = decoder(*fc, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
    fc_new = encoder(content_new)
    mse_loss = nn.MSELoss()
    loss_r = mse_loss(content_new, content)
    loss_p_list = []
    for i in range(5):
        loss_p_list.append(mse_loss(fc_new[i], fc[i]))
    loss_p = sum(loss_p_list) / len(loss_p_list)
    loss = 0.5 * loss_r + 0.5 * loss_p
    return loss

def get_dataloader(content_root):
    transferset = TransferDataset(content_root)
    loader = DataLoader(transferset, 8, True, num_workers=8, drop_last=True)
    return loader


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.95 ** epoch)


def show_results(content, style, no_train, train):
    plt.subplot(221)
    plt.imshow(content)
    plt.title('content')
    plt.subplot(222)
    plt.imshow(style)
    plt.title('style')
    plt.subplot(223)
    plt.imshow(no_train)
    plt.title('close_form')
    plt.subplot(224)
    plt.imshow(train)
    plt.title('close_form + train_decoder')
    plt.show()


def train_single_epoch(args, epoch, encoder, decoder, loader, optimizer, alpha_train=0):
    for i, (content_batch, style_batch) in enumerate(loader):
        content_batch.requires_grad = False

        d0_control = args.d_control[:5]
        d1_control = args.d_control[5: 8]
        d2_control = args.d_control[9: 16]
        d3_control = args.d_control[16: 23]
        d4_control = args.d_control[23: 28]
        d5_control = args.d_control[28: 32]
        d0_control = [int(i) for i in d0_control]
        d1_control = [int(i) for i in d1_control]
        d2_control = [int(i) for i in d2_control]
        d3_control = [int(i) for i in d3_control]
        d4_control = [int(i) for i in d4_control]
        d5_control = [int(i) for i in d5_control]

        if args.gpu is not None:
            content_batch = content_batch.cuda()
        loss = get_loss(encoder, decoder, content_batch, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if i % 100 == 0:
            print('epoch: %d | batch: %d | loss: %.4f' % (epoch, i, loss.cpu().data))

        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()


def train(args, encoder, decoder):
    MAX_EPOCH = args.max_epoch
    content_root = args.training_dataset

    for param in encoder.parameters():
        param.requires_grad = False

    decoder.train(), encoder.eval()
    loader = get_dataloader(content_root)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for i in range(MAX_EPOCH):
        train_single_epoch(args, i, encoder, decoder, loader, optimizer)
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict, '{:s}/decoder_epoch_{:d}.pth.tar'.format(args.save_dir, i + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0)
    parser.add_argument('-s', '--save_dir', default=os.path.join(abs_dir, 'trained_models_aaai'))
    parser.add_argument('-d', '--d_control', default='01010000000100000000000000001111')
    parser.add_argument('-me', '--max_epoch', default=2, type=int)
    parser.add_argument('-t', '--training_dataset')
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    net_e, net_d = load_nets()
    if args.gpu is not None:
        net_e.cuda()
        net_d.cuda()

    train(args, net_e, net_d)
    


