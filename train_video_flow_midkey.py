import csv
import random
import time
import argparse
import os

import cv2
import flow_vis
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from network import *
from utils import *
from metrics import PSNR, SSIM, MSE

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Configure
parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# Model
parser.add_argument('--in-features', type=int, default=3, metavar='N',
        help='the number of input features (default: 2)')
parser.add_argument('--out-features', type=int, default=5, metavar='N',
        help='the number of output features (default: 5)')
parser.add_argument('--hidden-features', type=int, default=256, metavar='N',
        help='the number of hidden units (default: 256)')
parser.add_argument('--hidden-layers', type=int, default=3, metavar='N',
        help='the number of layers (default: 3)')

# Training
parser.add_argument('--seed', type=int, default=50236, metavar='S',
        help='random seed (default: 50236)')
parser.add_argument('--lr', type=float, default=0.001, metavar='N',
        help='learning rate (default: 0.001)')
parser.add_argument('--training-step', type=int, default=30000, metavar='N',
        help='the number of training iterations (default: 30000)')
parser.add_argument('--flow-warmup-step', type=int, default=2000, metavar='N',
        help='flow only training warmup (default: 3000)')
parser.add_argument('--image-warmup-step', type=int, default=5000, metavar='N',
        help='flow only training warmup (default: 5000)')

parser.add_argument('--data-dir', type=str, default='./data', help='video name')
parser.add_argument('--video-name', type=str, default='alley_1', help='video name')
parser.add_argument('--start-frame', type=int, default=0, metavar='N',
        help='the starting frame (default: 0)')
parser.add_argument('--num-frames', type=int, default=7, metavar='N',
        help='the number of frames (default: 7)')
parser.add_argument('--jpeg-quality', type=int, default=90, metavar='N',
        help='jpeg quality (default: 90)')
parser.add_argument('--use-estimator', action='store_true', default=False)
parser.add_argument('--tag', type=str, default='temp', help='tag')
parser.add_argument('--checkpoint-iter', type=int, default=0, metavar='N', help='checkpoint iteration')


if __name__=='__main__':
    args = parser.parse_args()
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)

    target_frames, target_flow, key_frame, key_frame_size = load_video(
        args.data_dir, args.video_name, args.num_frames, args.start_frame,
        args.jpeg_quality, tag=args.tag, use_estimator=args.use_estimator)
    target_flow = target_flow.permute(0,2,3,1).cuda()
    target_frames = target_frames.cuda()
    key_frame = key_frame.cuda()
    T = args.num_frames
    H,W = target_frames.shape[2:]

    input_grid = make_input_grid(T, H, W)
    input_grid = input_grid.reshape(T, -1, args.in_features)
    flow_grid = make_flow_grid(H,W)
    flow_grid = flow_grid.unsqueeze(0)
    
    target_residual = []
    key_index = int((T/2))
    for t in reversed(range(key_index)):
        target_flow_grid_shift = apply_flow(flow_grid, target_flow[t].unsqueeze(0), H, W, direction='rl')
        warped_im_target = F.grid_sample(target_frames[t+1].unsqueeze(0),
                target_flow_grid_shift, padding_mode='border', align_corners=True)
        target_residual.insert(0,warped_im_target - target_frames[t])
    for t in range(key_index+1, T):
        target_flow_grid_shift = apply_flow(flow_grid, target_flow[t-1].unsqueeze(0), H, W, direction='lr')
        warped_im_target = F.grid_sample(target_frames[t-1].unsqueeze(0),
                target_flow_grid_shift, padding_mode='border', align_corners=True)
        target_residual.append(warped_im_target - target_frames[t])
    target_residual = torch.concat(target_residual, 0)

    net = Siren(hidden_features = args.hidden_features,
        hidden_layers = args.hidden_layers,
        in_features = args.in_features,
        out_features = args.out_features,
        outermost_linear = True)

    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = args.lr)

    total_num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_byte_size = total_num_params*4+key_frame_size

    metrics_list = []
    if args.checkpoint_iter > 0:
        dirname = "./results/{}/{}/".format(args.video_name, args.tag)
        model_path = dirname + "model_{:05d}.pt".format(args.checkpoint_iter)
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        metrics_list = np.loadtxt(dirname+"metrics.csv", delimiter=',').tolist()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

    start = time.time()
    metrics = [MSE(), PSNR(), SSIM()]
    max_iter = args.training_step + args.checkpoint_iter
    for epoch in range(args.checkpoint_iter, max_iter):
        net.train()
        optimizer.zero_grad()
        metric_epoch = [0.0]*(len(metrics)+2)
        for t in reversed(range(key_index)):
            flow, residual = net(input_grid[t])
            flow = flow.reshape(1,H,W,2)
            residual = residual.permute(1,0).reshape(1,3,H,W)

            # making flow grid for grid_sample
            # we are sampling pixels from the previous frame (right -> left)
            flow_grid_shift = apply_flow(flow_grid, flow, H, W, direction='rl')

            # warping image based on reconstred image or ground truth image
            if epoch <= args.image_warmup_step: 
                if t==key_index-1:
                    warped_im = F.grid_sample(key_frame.unsqueeze(0), flow_grid_shift, padding_mode='border', align_corners=True)
                else:
                    warped_im = F.grid_sample(target_frames[t+1].unsqueeze(0), flow_grid_shift, padding_mode='border', align_corners=True)
            else:
                if t==key_index-1:
                    warped_im = F.grid_sample(key_frame.unsqueeze(0), flow_grid_shift, padding_mode='border', align_corners=True)
                else:
                    warped_im = F.grid_sample(reconstructed_im.detach(), flow_grid_shift, padding_mode='border', align_corners=True)
            reconstructed_im = warped_im + residual

            # metrics
            for i in range(len(metrics)):
                metric_epoch[i]+=metrics[i](reconstructed_im, target_frames[t].unsqueeze(0)).item()

            # reconstructed error
            im_error = reconstructed_im - target_frames[t].unsqueeze(0)
            im_loss = torch.norm(im_error)

            # flow error
            flow_error = flow - target_flow[t].unsqueeze(0)
            metric_epoch[-2] += float(torch.mean(flow_error.detach()**2))
            flow_loss = torch.norm(flow_error)
            # residual error
            residual_error = residual - target_residual[t].unsqueeze(0)
            metric_epoch[-1] += float(torch.mean(residual_error.detach()**2))
            residual_loss = torch.norm(residual_error)

            if epoch <= args.flow_warmup_step:
                loss = flow_loss+residual_loss
            else:
                loss = im_loss
            loss.backward()

        for t in range(key_index+1, T):
            flow, residual = net(input_grid[t-1])
            flow = flow.reshape(1,H,W,2)
            residual = residual.permute(1,0).reshape(1,3,H,W)

            # making flow grid for grid_sample
            # we are sampling pixels from the previous frame (left -> right)
            flow_grid_shift = apply_flow(flow_grid, flow, H, W, direction='lr')

            # warping image based on reconstred image or ground truth image
            if epoch <= args.image_warmup_step:
                if t==key_index+1:
                    warped_im = F.grid_sample(key_frame.unsqueeze(0), flow_grid_shift, padding_mode='border', align_corners=True)
                else:
                    warped_im = F.grid_sample(target_frames[t-1].unsqueeze(0), flow_grid_shift, padding_mode='border', align_corners=True)
            else:
                if t==key_index+1:
                    warped_im = F.grid_sample(key_frame.unsqueeze(0), flow_grid_shift, padding_mode='border', align_corners=True)
                else:
                    warped_im = F.grid_sample(reconstructed_im.detach(), flow_grid_shift, padding_mode='border', align_corners=True)
            reconstructed_im = warped_im + residual

            # metrics
            for i in range(len(metrics)):
                metric_epoch[i]+=metrics[i](reconstructed_im, target_frames[t].unsqueeze(0)).item()

            # reconstructed error
            im_error = reconstructed_im - target_frames[t].unsqueeze(0)
            im_loss = torch.norm(im_error)

            # flow error
            flow_error = flow - target_flow[t-1].unsqueeze(0)
            metric_epoch[-2] += float(torch.mean(flow_error.detach()**2))
            flow_loss = torch.norm(flow_error)
            # residual error
            residual_error = residual - target_residual[t-1].unsqueeze(0)
            metric_epoch[-1] += float(torch.mean(residual_error.detach()**2))
            residual_loss = torch.norm(residual_error)

            if epoch <= args.flow_warmup_step:
                loss = flow_loss+residual_loss
            else:
                loss = im_loss
            loss.backward()
            
        # updating after entire frames
        optimizer.step()

        # logging
        net.eval()
        if epoch % 50 == 0:
            for i in range(len(metrics)):
                metric_epoch[i]+=metrics[i](key_frame.unsqueeze(0), target_frames[key_index].unsqueeze(0)).item()
            metric_epoch = [m/T for m in metric_epoch]
            metrics_list.append([model_byte_size]+metric_epoch)
            print("[epoch {}] Size(bytes): {}, [MSE, PSNR, SSIM, MSE(flow), MSE(residual)]: [{:.06f}, {:.06f}, {:.06f}, {:.06f}, {:.06f}]".format(
                    epoch, model_byte_size, metric_epoch[0], metric_epoch[1], metric_epoch[2], metric_epoch[3], metric_epoch[4]))

        if epoch % 5000 == 0:
            dirname = "./results/{}/{}/".format(args.video_name, args.tag)
            epoch_dirname = dirname + "{}_{}/".format(args.tag, epoch) 
            if not os.path.exists(epoch_dirname):
                os.makedirs(epoch_dirname)
            # from left to key frame
            for t in reversed(range(key_index)):
                pred_flow, residual = net(input_grid[t])
                pred_flow = pred_flow.reshape(1,H,W,2)
                residual = residual.permute(1,0).reshape(1,3,H,W)
                
                flow_grid_shift = apply_flow(flow_grid, pred_flow, H, W, direction='rl')
                target_flow_grid_shift = apply_flow(flow_grid, target_flow[t].unsqueeze(0), H, W, direction='rl')
                # using key frames
                if t==key_index-1:
                    warped_im = F.grid_sample(target_frames[t+1].unsqueeze(0), flow_grid_shift, padding_mode='border', align_corners=True)
                else:
                    warped_im = F.grid_sample(reconstructed_im, flow_grid_shift, padding_mode='border', align_corners=True)
                warped_im_target = F.grid_sample(target_frames[t+1].unsqueeze(0), target_flow_grid_shift, padding_mode='border', align_corners=True)
                reconstructed_im = (warped_im + residual).detach()
                pred_flow_im = flow2img(pred_flow.detach().squeeze().cpu().numpy())
                target_flow_im = flow2img(target_flow[t].detach().squeeze().cpu().numpy())
                residual_im = torch.abs(residual)
                residual_im_target = torch.abs((warped_im_target - target_frames[t]))

                show_tensor_to_image(warped_im.reshape(3, H, W), epoch_dirname+"warped_im_{:05d}".format(t))
                show_tensor_to_image(residual_im.reshape(3, H, W), epoch_dirname+"residual_im_{:05d}".format(t))
                show_tensor_to_image(reconstructed_im.reshape(3, H, W), epoch_dirname+"reconstructed_im_{:05d}".format(t))
                show_tensor_to_image(torch.Tensor(pred_flow_im/255).permute(2,0,1), epoch_dirname+"/pred_flow_{:05d}".format(t))
                show_tensor_to_image(warped_im_target.reshape(3, H, W), epoch_dirname+"warped_im_target_{:05d}".format(t))
                show_tensor_to_image(residual_im_target.reshape(3, H, W), epoch_dirname+"residual_im_target{:05d}".format(t))
                show_tensor_to_image(torch.Tensor(target_flow_im/255).permute(2,0,1), epoch_dirname+"/target_flow_{:05d}".format(t))
                show_tensor_to_image(target_frames[t], epoch_dirname+"target_frame_{:05d}".format(t))
            # to store key frame
            target_flow_im = flow2img(target_flow[key_index].detach().squeeze().cpu().numpy())
            show_tensor_to_image(torch.Tensor(target_flow_im/255).permute(2,0,1), epoch_dirname+"/target_flow_{:05d}".format(key_index))
            show_tensor_to_image(target_frames[key_index], epoch_dirname+"target_frame_{:05d}".format(key_index))
            # from key frame to right
            for t in range(key_index+1, T):
                pred_flow, residual = net(input_grid[t-1])
                pred_flow = pred_flow.reshape(1,H,W,2)
                residual = residual.permute(1,0).reshape(1,3,H,W)
                
                flow_grid_shift = apply_flow(flow_grid, pred_flow, H, W, direction='lr')
                target_flow_grid_shift = apply_flow(flow_grid, target_flow[t-1].unsqueeze(0), H, W, direction='lr')
                # using key frames
                if t==key_index+1:
                    warped_im = F.grid_sample(target_frames[t-1].unsqueeze(0), flow_grid_shift, padding_mode='border', align_corners=True)
                else:
                    warped_im = F.grid_sample(reconstructed_im, flow_grid_shift, padding_mode='border', align_corners=True)
                warped_im_target = F.grid_sample(target_frames[t-1].unsqueeze(0), target_flow_grid_shift, padding_mode='border', align_corners=True)
                reconstructed_im = (warped_im + residual).detach()
                pred_flow_im = flow2img(pred_flow.detach().squeeze().cpu().numpy())
                target_flow_im = flow2img(target_flow[t-1].detach().squeeze().cpu().numpy())
                residual_im = torch.abs(residual)
                residual_im_target = torch.abs((warped_im_target - target_frames[t]))

                show_tensor_to_image(warped_im.reshape(3, H, W), epoch_dirname+"warped_im_{:05d}".format(t))
                show_tensor_to_image(residual_im.reshape(3, H, W), epoch_dirname+"residual_im_{:05d}".format(t))
                show_tensor_to_image(reconstructed_im.reshape(3, H, W), epoch_dirname+"reconstructed_im_{:05d}".format(t))
                show_tensor_to_image(torch.Tensor(pred_flow_im/255).permute(2,0,1), epoch_dirname+"/pred_flow_{:05d}".format(t))
                show_tensor_to_image(warped_im_target.reshape(3, H, W), epoch_dirname+"warped_im_target_{:05d}".format(t))
                show_tensor_to_image(residual_im_target.reshape(3, H, W), epoch_dirname+"residual_im_target{:05d}".format(t))
                show_tensor_to_image(torch.Tensor(target_flow_im/255).permute(2,0,1), epoch_dirname+"/target_flow_{:05d}".format(t))
                show_tensor_to_image(target_frames[t], epoch_dirname+"target_frame_{:05d}".format(t))

        if (epoch+1) % 10000 == 0:
            model_path = dirname + "model_{:05d}.pt".format(epoch+1)
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, model_path)
            np.savetxt(dirname+"metrics.csv", metrics_list, delimiter=",")

    end = time.time()
    print("training_time: {}".format(end - start))

