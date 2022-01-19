import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from flow_utils import *
from metrics import PSNR, SSIM, MSE
from network import *
from utils import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='temp', help='tag (name)')
parser.add_argument('--save_path', type=str, default='./')

# model
parser.add_argument('--hidden_features', type=int, default=96,
                    help='the number of hidden units per layer')
parser.add_argument('--hidden_layers', type=int, default=3,
                    help='the number of layers (default: 3)')

# video
parser.add_argument('--video_path', type=str, default='./training/final/alley_1',
                    help='video path (for images use folder path, '
                         'for a video use video path)')
parser.add_argument('--video_scale', type=float, default=0.5,
                    help='the height and width of a output video will be '
                         'multiplied by video_scale')
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--n_frames', type=int, default=7,
                    help='the number of frames')

# training
parser.add_argument('--seed', type=int, default=50236, metavar='S',
                    help='random seed (default: 50236)')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--epochs', type=int, default=30000,
                    help='the number of training epochs')

parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--save_logs_interval', type=int, default=10000)


if __name__=='__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    print('configs')
    print(str(vars(args)))

    """ PREPARING A VIDEO """
    target_frames = load_video_from_images(
        args.video_path, args.start_frame, args.n_frames, args.video_scale)
    target_frames = target_frames.cuda()

    # grids
    T, _, H, W = target_frames.size()
    input_grid = make_input_grid(T, H, W)

    """ PREPARING A NETWORK """
    net = Siren(in_features=3,
                hidden_features=args.hidden_features,
                hidden_layers=args.hidden_layers,
                out_features=3,
                outermost_linear=True)
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    model_size = sum(p.numel() for p in net.parameters()) * 4
    print(f'total bytes ({model_size}) = model size ({model_size})')

    """ MISC """
    metrics = [PSNR(), SSIM()]
    n_metrics = len(metrics)

    header = ', '.join([str(m) for m in metrics])
    perf_logs = []

    save_path = os.path.join(args.save_path, args.tag)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """ START TRAINING """
    with tqdm.tqdm(range(args.epochs)) as loop:
        for epoch in loop:
            net.train()
            optimizer.zero_grad()

            is_eval_epoch = (epoch + 1) % args.eval_interval == 0

            if is_eval_epoch:
                perf_logs.append(np.zeros(n_metrics))

            for i in range(T):
                outputs = net(input_grid[i])
                outputs = torch.sigmoid(outputs)
                outputs = outputs.permute(2, 0, 1) # RGB
                
                loss = F.mse_loss(outputs, target_frames[i])
                loss.backward()

            # update per epoch
            optimizer.step()

            if is_eval_epoch:
                net.eval()

                for i in range(T):
                    outputs = net(input_grid[i])
                    outputs = torch.sigmoid(outputs)
                    outputs = outputs.permute(2, 0, 1).unsqueeze(0) # RGB

                    for j in range(n_metrics):
                        perf_logs[-1][j] += metrics[j](
                            outputs,
                            target_frames[i].unsqueeze(0)).item() / T

                postfix = {str(metrics[i]): perf_logs[-1][i]
                           for i in range(n_metrics)} # test performance
                loop.set_postfix(postfix)

            # TODO: visualize

            # save logs
            if (epoch+1) % args.save_logs_interval == 0:
                model_path = os.path.join(save_path, f'model_{epoch+1:05d}.pt')
                torch.save({'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, model_path)
                np.savetxt(os.path.join(save_path, "logs.csv"),
                           perf_logs, fmt='%0.6f',
                           delimiter=", ", header=header, comments='')

