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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=1)
    parser.add_argument('-sd', '--save_dir', default=os.path.join(abs_dir, 'result_val_nas'))
    parser.add_argument('-c', '--content', default='/mnt/home/xiaoxiang/haozhe/style_nas_2/content')
    parser.add_argument('-s', '--style', default='/mnt/home/xiaoxiang/haozhe/style_nas_2/style')
    parser.add_argument('-a', '--alpha', default=1.0)
    parser.add_argument('-d', '--d_control')
    args = parser.parse_args()
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    net_e, net_d0, net_d1, net_d2, net_d3, net_d4, net_d5 = load_net()
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
        net_e.cuda(), net_e.eval() 
        net_d0.cuda(), net_d0.eval()
        net_d1.cuda(), net_d1.eval()
        net_d2.cuda(), net_d2.eval()
        net_d3.cuda(), net_d3.eval()
        net_d4.cuda(), net_d4.eval()
        net_d5.cuda(), net_d5.eval()

    content_list = get_test_list(args.content)
    style_list = get_test_list(args.style)
    content_list = [i for i in content_list if '.jpg' in i]
    style_list = [i for i in style_list if '.jpg' in i]
    content_list.sort()
    style_list.sort()
    for k in range(len(style_list[:])):
        content_path = content_list[k]
        style_path = style_list[k]
        print('----- transfering pair %d -------' % (k))
        content = get_a_image(content_path)        
        style = get_a_image(style_path)        
        content_save = content
        style_save = style
        content, style = resize_imgs(content, style)
        content = transforms.ToTensor()(content)
        style = transforms.ToTensor()(style)
        content = content.unsqueeze(0)
        style = style.unsqueeze(0)
        if args.gpu is not None:
            content = content.cuda()
            style = style.cuda()
        cF = list(net_e(content))
        sF = list(net_e(style))
        csF = []
        for ii in range(len(cF)):
            if ii == 0:
                if d0_control[0] == 1:
                    this_csF = transform(cF[ii], sF[ii], args.alpha)
                    csF.append(this_csF)
                else:
                    csF.append(cF[ii])
            elif ii == 1:
                if d2_control[-1] == 1:
                    this_csF = transform(cF[ii], sF[ii], args.alpha)
                    csF.append(this_csF)
                else:
                    csF.append(cF[ii])
            elif ii == 2:
                if d3_control[-1] == 1:
                    this_csF = transform(cF[ii], sF[ii], args.alpha)
                    csF.append(this_csF)
                else:
                    csF.append(cF[ii])
            elif ii == 3:
                if d4_control[-1] == 1:
                    this_csF = transform(cF[ii], sF[ii], args.alpha)
                    csF.append(this_csF)
                else:
                    csF.append(cF[ii])
            elif ii == 4:
                if d5_control[-1] == 1:
                    this_csF = transform(cF[ii], sF[ii], args.alpha)
                    csF.append(this_csF)
                else:
                    csF.append(cF[ii])
            else:
                csF.append(cF[ii])

        csF[0] = net_d0(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d0(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if d1_control[0] == 1:
            csF[0] = transform(csF[0], sF[0], args.alpha)
        csF[0] = net_d1(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d1(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if d2_control[0] == 1:
            csF[0] = transform(csF[0], sF[0], args.alpha)
        csF[0] = net_d2(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d2(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if d3_control[0] == 1:
            csF[0] = transform(csF[0], sF[0], args.alpha)
        csF[0] = net_d3(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d3(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if d4_control[0] == 1:
            csF[0] = transform(csF[0], sF[0], args.alpha)
        csF[0] = net_d4(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d4(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if d5_control[0] == 1:
            csF[0] = transform(csF[0], sF[0], args.alpha)
        csF[0] = net_d5(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d5(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        out = csF[0].cpu().data.float()
        utils.save_image(out, os.path.join(args.save_dir, '%d.jpg' % (k)))
        out = cv2.imread(os.path.join(args.save_dir, '%d.jpg' % (k)))
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        content_save, style_save, out = resize_save(content_save, style_save, out)
        # out_compare = np.concatenate((content_save, style_save, out), 1)
        cv2.imwrite(os.path.join(args.save_dir, '%d.jpg' % (k)), out)


