import csv
import random
import time
import argparse
import os

import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from network import *
from utils import *

from metrics import PSNR, SSIM, MSE

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Configure
parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# Model
parser.add_argument('--in-features', type=int, default=3, metavar='N',
        help='the number of input features (default: 3)')
parser.add_argument('--out-features', type=int, default=3, metavar='N',
        help='the number of output features (default: 3)')
parser.add_argument('--hidden-features', type=int, default=512, metavar='N',
        help='the number of hidden units (default: 512)')
parser.add_argument('--hidden-layers', type=int, default=5, metavar='N',
        help='the number of layers (default: 5)')
parser.add_argument('--embedding-size', type=int, default=128, metavar='N',
        help='feature dimension (default: 128)')
parser.add_argument('--activation', default='relu', type=str,
        help='activation function')

# Training
parser.add_argument('--seed', type=int, default=50236, metavar='S',
        help='random seed (default: 50236)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
        help='learning rate (default: 0.0001)')
parser.add_argument('--training-step', type=int, default=10000, metavar='N',
        help='the number of training iterations (default: 10000)')

parser.add_argument('--data-dir', type=str, default='./data', help='video name')
parser.add_argument('--video-name', type=str, default='alley_1', help='video name')
parser.add_argument('--num-frames', type=int, default=10, metavar='N',
        help='the number of layers (default: 10)')
parser.add_argument('--tag', type=str, default='temp', help='tag')


if __name__=='__main__':
    args = parser.parse_args()
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)

    tf = transforms.Compose([
        transforms.ToTensor(),
        #transforms.CenterCrop(384),
        transforms.Resize((218, 512)), # (1024, 436)
    ])

    video_dir = args.data_dir + "/final/{}/".format(args.video_name)
    target_frames = []
    for t in range(args.num_frames):
        target_frames.append(tf(PIL.Image.open(video_dir+"frame_{:04d}.png".format(t+1))))
    target_frames = torch.stack(target_frames, 0).cuda()
    T = args.num_frames
    H,W = target_frames.shape[2:]

    target_frames = target_frames.permute(0, 2, 3, 1).reshape(-1, H*W, 3)
    input_grid = make_input_grid(T, H, W).cuda()
    input_grid = input_grid.reshape(T, -1, args.in_features)
    
    net = Siren(hidden_features = args.hidden_features,
        hidden_layers = args.hidden_layers,
        in_features = args.in_features,
        out_features = args.out_features,
        outermost_linear = True,
        flow=False)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = args.lr)

    total_num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_byte_size = total_num_params*4

    start = time.time()
    metrics = [MSE(), PSNR(), SSIM()]
    metrics_list = []
    for epoch in range(args.training_step):
        net.eval()
        optimizer.zero_grad()
        metric_epoch = [0.0]*len(metrics)
        for t in range(T):
            grid_chunk = input_grid[t,:,:].unsqueeze(0)
            target_chunk = target_frames[t,:,:].unsqueeze(0)
            pred = net(grid_chunk)
            # metrics
            for i in range(len(metrics)):
                metric_epoch[i]+=metrics[i](pred.permute(0,2,1).reshape(1,3,H,W),
                                target_chunk.permute(0,2,1).reshape(1,3,H,W)).item()
            im_error = pred - target_chunk
            loss = torch.norm(im_error)
            loss.backward()
        optimizer.step()

        # logging
        net.eval()
        if epoch % 50 == 0:
            metric_epoch = [m/T for m in metric_epoch]
            metrics_list.append(metric_epoch)
            print("[epoch {}] Size(bytes): {}, MSE: {}, PSNR: {}, SSIM: {}".format(
                    epoch, model_byte_size, metric_epoch[0], metric_epoch[1], metric_epoch[2]))

        if epoch % 5000 == 0:
            dirname = "./results/{}/{}/".format(args.video_name, args.tag)
            epoch_dirname = dirname + "{}_{}/".format(args.tag, epoch) 
            if not os.path.exists(epoch_dirname):
                os.makedirs(epoch_dirname)
            np.savetxt(dirname+"metrics.csv", metrics_list, delimiter=",")
            for t in range(T):
                pred = net(input_grid[t,:,:])
                path = epoch_dirname+"/pred_{:05d}".format(t)
                show_tensor_to_image(pred.permute(1, 0).reshape(-1, H, W), path)
        
        if (epoch+1) % 5000 == 0:
            model_path = dirname + "model_{:05d}.pt".format(epoch+1)
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimiaer_state_dict': optimizer.state_dict(),
                        }, model_path)

    end = time.time()
    print("training_time: {}".format(end - start))

