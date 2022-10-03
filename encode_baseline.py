import argparse
import csv
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torchvision.utils import save_image
from sympy.solvers import solve
from sympy import Symbol

from embedding import *
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
parser.add_argument('--bpp', type=float, default=0.1)
parser.add_argument('--use_amp', action='store_true',
                    help='use automatic mixed precision (32 to 16 bits)')

# video
parser.add_argument('--video', type=str,
                    default='./training/final/alley_1',
                    help='video path (for images use folder path, '
                         'for a video use video path)')
parser.add_argument('--video_scale', type=float, default=1,
                    help='the height and width of a output video will be '
                         'multiplied by video_scale')
parser.add_argument('--n_frames', type=int, default=500, # 15,
                    help='the number of frames')
parser.add_argument('--max_frames', type=int, default=None)

# training
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=5000,
                    help='the number of training epochs')

parser.add_argument('--eval_interval', type=int, default=None)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--visualize', action='store_true')


class EmptyContext:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def main(args, target_frames, metrics,
         save_path, name='', verbose=False, visualize=True):
    target_frames = target_frames.cuda()

    # grids
    T, _, H, W = target_frames.size()
    input_grid = make_input_grid(T, H, W)

    """ PREPARING A NETWORK """
    x = Symbol('x')
    eq = 3*x**2 + 10*x + 3 - args.bpp*T*H*W/8/(4 - 2*args.use_amp)
    width = int(np.round(float(max(solve(eq)))/2)*2)

    net = Siren(in_features=3,
                hidden_features=width,
                hidden_layers=3,
                out_features=3,
                outermost_linear=True)
    net = nn.DataParallel(net.cuda())

    optimizer = optim.Adam(net.parameters(),
                           betas=(0.9, 0.99), eps=1e-15, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    total_size = sum(p.numel() for p in net.parameters()) * (4-2*args.use_amp)

    """ MISC """
    n_metrics = len(metrics)

    header = ', '.join([str(m) for m in metrics] \
                       + ['MSE(flows)', 'MSE(residuals)'])
    perf_logs = []

    print(f'approx. total bytes ({total_size}({width}))')
    if verbose:
        print(net)

    if args.use_amp:
        context = lambda : torch.cuda.amp.autocast(enabled=True)
    else:
        context = EmptyContext

    """ START TRAINING """
    net.train()
    with EmptyContext(): # tqdm.tqdm(range(args.epochs)) as loop:
        for epoch in range(args.epochs):
            optimizer.zero_grad()

            is_eval_epoch = (epoch + 1) % args.eval_interval == 0

            if is_eval_epoch:
                perf_logs.append(np.zeros(n_metrics))

            for i in np.random.permutation(range(T)):
                with context():
                    i_grid = input_grid[i] # [H, W, 3]
                    target = target_frames[i].unsqueeze(0)

                    outputs = net(i_grid)
                    outputs = outputs.permute(2, 0, 1).unsqueeze(0)

                    loss = F.mse_loss(outputs, target)

                    if args.use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            # update
            if args.use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()

            if is_eval_epoch:
                net.eval()

                total_time = 0

                for i in range(T):
                    with torch.no_grad():
                        with context():
                            start = time.time()
                            outputs = net(input_grid[i])
                            outputs = outputs.permute(2, 0, 1).unsqueeze(0)
                            total_time += time.time() - start

                    if visualize:
                        postfix = f'{name}_{epoch:04d}:{dst}.jpg'
                        save_image(outputs[0],
                                   os.path.join(save_path, f'final_{postfix}'))
                        save_image(target_frames[dst],
                                   os.path.join(save_path,
                                                f'target_{name}:{dst}.jpg'))

                    for j in range(n_metrics):
                        value = metrics[j](
                            outputs.float(),
                            target_frames[i].unsqueeze(0)).item()
                        perf_logs[-1][j] += value / T

                net.train()

                # print(f'FPS: {T / total_time}')
                # postfix = {str(metrics[i]): perf_logs[-1][i]
                #            for i in range(n_metrics)} # test performance
                # loop.set_postfix(postfix)

            # save logs
            if (epoch+1) == args.epochs:
                model_path = os.path.join(
                    save_path, f'model_{name}_{epoch+1:05d}.pt')
                torch.save({'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, model_path)
                np.savetxt(os.path.join(save_path, "logs.csv"),
                           perf_logs, fmt='%0.6f',
                           delimiter=", ", header=header, comments='')

    return perf_logs[-1], total_size


if __name__ == '__main__':
    args = parser.parse_args()

    if args.eval_interval is None:
        args.eval_interval = args.epochs

    print('configs')
    print(str(vars(args)))

    save_path = os.path.join(args.save_path, args.tag)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """ PREPARING A VIDEO """
    target_frames = load_video(args.video, scale=args.video_scale)
    if args.max_frames is not None:
        target_frames = target_frames[:args.max_frames]
    total_frames = len(target_frames)
    print(target_frames.shape)

    n_gop = int(max(np.round(total_frames / args.n_frames), 1))
    gop_indices = np.round(np.linspace(0, total_frames, n_gop+1)).astype('int')

    # keyframes
    metrics = [PSNR(), SSIM()]

    # compress
    performances = []
    model_sizes = []
    start = time.time()

    for i in range(n_gop): 
        # print(gop_indices[i:i+2])
        perfs, size = main(
            args,
            target_frames[gop_indices[i]:gop_indices[i+1]],
            metrics,
            save_path, f'{i}', args.verbose, args.visualize)
        performances.append(perfs)
        model_sizes.append(size)

    total_size = sum(model_sizes)

    performances = np.stack(performances, 0)
    weights = ((gop_indices[1:] - gop_indices[:-1])
               / total_frames)[..., None]
    performances = np.sum(performances * weights, 0)

    print(f'total performances: {performances} ({time.time() - start} sec)')
    print(f'total_size: {total_size}')
    print(f'bpp: {8*total_size/(total_frames * target_frames.shape[-2] * target_frames.shape[-1])}')

