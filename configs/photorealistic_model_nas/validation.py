import sys
sys.path.insert(0, '/mnt/home/xiaoxiang/haozhe/style_nas_2/models/')
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from dataset import TransferDataset
from torch.utils.serialization import load_lua
import argparse
from torchvision import utils
from torchvision import transforms
from models_photorealistic_nas.VGG_with_decoder import encoder, decoder0, decoder1, decoder2, decoder3, decoder4, decoder5
from wct import transform

abs_dir = os.path.abspath(os.path.dirname(__file__))

def load_net():
    encoder_param = load_lua('/mnt/home/xiaoxiang/haozhe/style_nas_2/models/models_photorealistic_nas/vgg_normalised_conv5_1.t7')
    net_e = encoder(encoder_param)
    net_d0 = decoder0()
    net_d0.load_state_dict(torch.load(os.path.join(abs_dir, 'trained_models_nas/decoder_epoch_2.pth.tar')))
    net_d1 = decoder1()
    net_d1.load_state_dict(torch.load(os.path.join(abs_dir, 'trained_models_nas/decoder_epoch_2.pth.tar')))
    net_d2 = decoder2()
    net_d2.load_state_dict(torch.load(os.path.join(abs_dir, 'trained_models_nas/decoder_epoch_2.pth.tar')))
    net_d3 = decoder3()
    net_d3.load_state_dict(torch.load(os.path.join(abs_dir, 'trained_models_nas/decoder_epoch_2.pth.tar')))
    net_d4 = decoder4()
    net_d4.load_state_dict(torch.load(os.path.join(abs_dir, 'trained_models_nas/decoder_epoch_2.pth.tar')))
    net_d5 = decoder5()
    net_d5.load_state_dict(torch.load(os.path.join(abs_dir, 'trained_models_nas/decoder_epoch_2.pth.tar')))
    return net_e, net_d0, net_d1, net_d2, net_d3, net_d4, net_d5

def get_test_list(root_dir):
    test_list = os.listdir(root_dir)
    test_list = [os.path.join(root_dir, i) for i in test_list]
    return test_list

def get_a_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def resize_save(content, style, out):
    if content.shape[0] < content.shape[1]:
        out_h = 512
        out_w = np.int32(512.0 * content.shape[1] / content.shape[0])
    else:
        out_w = 512
        out_h = np.int32(512.0 * content.shape[0] / content.shape[1])
    content = cv2.resize(content, (out_w, out_h), cv2.INTER_AREA)
    style = cv2.resize(style, (out_w, out_h), cv2.INTER_AREA)
    out = cv2.resize(out, (out_w, out_h), cv2.INTER_AREA)
    return content, style, out

def resize_imgs(content, style):
    c_h = 384
    c_w = 768
    s_h = 384
    s_w = 768
    # c_h = content.shape[0]
    # c_w = content.shape[1]
    # s_h = style.shape[0]
    # s_w = style.shape[1]
    # c_ratio = np.float32(c_h) / c_w
    # s_ratio = np.float32(s_h) / s_w
    # if (c_ratio / s_ratio > 4.0) or (s_ratio / c_ratio > 4.0):
        # c_h_out = 512
        # c_w_out = 512
        # s_h_out = 512
        # s_w_out = 512
    # elif c_ratio < 1:
        # c_h = 512
        # c_w = np.int32(c_h / c_ratio)
        # s_h = c_h
        # s_w = c_w
    # elif c_ratio >= 1:
        # c_w = 512
        # c_h = np.int32(c_w * c_ratio)
        # s_h = c_h
        # s_w = c_w
    content = cv2.resize(content, (c_w, c_h), cv2.INTER_AREA)
    style = cv2.resize(style, (s_w, s_h), cv2.INTER_AREA)
    return content, style

def handmake_mse(result, target):
    return torch.mean((result - target) ** 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0)
    args = parser.parse_args()

    net_e, _, _, _, _, _, _ = load_net()

    if args.gpu is not None:
        net_e.cuda(), net_e.eval() 

    validation_list = get_test_list(os.path.join(abs_dir, 'result_val_nas'))
    benchmark_list = get_test_list('/mnt/home/xiaoxiang/haozhe/style_nas_2/benchmark')
    validation_list = [i for i in validation_list if '.jpg' in i]
    benchmark_list = [i for i in benchmark_list if '.jpg' in i]
    validation_list.sort()
    benchmark_list.sort()
    mse_loss = nn.MSELoss()
    loss_r_list = []
    loss_p_list = []
    for k in range(len(validation_list[:])):
        validation_path = validation_list[k]
        benchmark_path = benchmark_list[k]
        print('----- validate pair %d -------' % (k))
        validation_img = get_a_image(validation_path)        
        benchmark_img = get_a_image(benchmark_path)        
        validation_img, benchmark_img = resize_imgs(validation_img, benchmark_img)
        validation_img = transforms.ToTensor()(validation_img)
        benchmark_img = transforms.ToTensor()(benchmark_img)
        validation_img = validation_img.unsqueeze(0)
        benchmark_img = benchmark_img.unsqueeze(0)
        if args.gpu is not None:
            validation_img = validation_img.cuda()
            benchmark_img = benchmark_img.cuda()
        validation_f = list(net_e(validation_img))
        benchmark_f = list(net_e(benchmark_img))
        loss_r = handmake_mse(validation_img, benchmark_img).cpu().data.numpy()
        loss_p_list = []
        for i in range(5):
            loss_p_list.append(handmake_mse(validation_f[i], benchmark_f[i]).cpu().data.numpy())
        loss_p = sum(loss_p_list) / len(loss_p_list)
        loss_r_list.append(loss_r)
        loss_p_list.append(loss_p)

    overall_loss_r = '%.4f' % (sum(loss_r_list) / len(loss_r_list))
    overall_loss_p = '%.4f' % (sum(loss_p_list) / len(loss_p_list))
    with open(os.path.join(abs_dir, 'result.txt'), 'w') as f:
        f.write('%s %s' % (overall_loss_r, overall_loss_p))

        


