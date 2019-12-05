import sys
sys.path.insert(0, '/gpfs/share/home/1601210097/projects/style_transfer/content_style_separation/')
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
from models_anti_multi_level_pyramid_stage_decoder_in.VGG_with_decoder import encoder, decoder0, decoder1, decoder2, decoder3, decoder4, decoder5

def load_net():
    encoder_param = load_lua('../../models_anti_multi_level_pyramid_stage_decoder_in/vgg_normalised_conv5_1.t7')
    net_e = encoder(encoder_param)
    net_d0 = decoder0()
    net_d0.load_state_dict(torch.load('./trained_models_anti_multi_level/decoder_epoch_5.pth.tar'))
    net_d1 = decoder1()
    net_d1.load_state_dict(torch.load('./trained_models_anti_multi_level/decoder_epoch_5.pth.tar'))
    net_d2 = decoder2()
    net_d2.load_state_dict(torch.load('./trained_models_anti_multi_level/decoder_epoch_5.pth.tar'))
    net_d3 = decoder3()
    net_d3.load_state_dict(torch.load('./trained_models_anti_multi_level/decoder_epoch_5.pth.tar'))
    net_d4 = decoder4()
    net_d4.load_state_dict(torch.load('./trained_models_anti_multi_level/decoder_epoch_5.pth.tar'))
    net_d5 = decoder5()
    net_d5.load_state_dict(torch.load('./trained_models_anti_multi_level/decoder_epoch_5.pth.tar'))
    return net_e, net_d0, net_d1, net_d2, net_d3, net_d4, net_d5

def get_test_list(root_dir):
    test_list = os.listdir(root_dir)
    test_list = [os.path.join(root_dir, i) for i in test_list]
    return test_list

def get_a_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0)
    parser.add_argument('-s', '--save_dir', default='./test_results')
    parser.add_argument('-c', '--content', default='/gpfs/share/home/1601210097/projects/style_transfer/content_images')
    args = parser.parse_args()
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    net_e, net_d0, net_d1, net_d2, net_d3, net_d4, net_d5 = load_net()
    if args.gpu is not None:
        net_e.cuda(), net_e.eval() 
        net_d0.cuda(), net_d0.eval()
        net_d1.cuda(), net_d1.eval()
        net_d2.cuda(), net_d2.eval()
        net_d3.cuda(), net_d3.eval()
        net_d4.cuda(), net_d4.eval()
        net_d5.cuda(), net_d5.eval()

    test_img_list = get_test_list(args.content)
    for i, img_path in enumerate(test_img_list):
        print('----- testing img %d -------' % i)
        img_save_in = get_a_image(img_path)        
        img_save_in = cv2.resize(img_save_in, (512, 512), cv2.INTER_AREA)
        img = transforms.ToTensor()(img_save_in)
        img = img.unsqueeze(0)
        if args.gpu is not None:
            img = img.cuda()
        features = list(net_e(img))
        features[0] = net_d0(*features)
        features[0] = net_d1(*features)
        features[0] = net_d2(*features)
        features[0] = net_d3(*features)
        features[0] = net_d4(*features)
        features[0] = net_d5(*features)
        out = features[0].cpu().data.float()
        utils.save_image(out, os.path.join(args.save_dir, 'comp_%d.jpg' % i))
        out = cv2.imread(os.path.join(args.save_dir, 'comp_%d.jpg' % i))
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out_compare = np.concatenate((img_save_in, out), 1)
        cv2.imwrite(os.path.join(args.save_dir, 'comp_%d.jpg' % i), out_compare)


