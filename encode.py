import argparse
import csv
import matplotlib.pyplot as plt
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


parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='temp', help='tag (name)')
parser.add_argument('--save_path', type=str, default='./')

# model
parser.add_argument('--backbone', type=str, default='nerf',
                    choices=['nerf', 'siren'])
parser.add_argument('--ratio', type=float, default=1,
                    help='the ratio between network and keyframe')
parser.add_argument('--hidden_layers', type=int, default=1,
                    help='the number of layers (default: 1)')
parser.add_argument('--reuse', type=int, default=2)
parser.add_argument('--use_amp', action='store_true',
                    help='use automatic mixed precision (32 to 16 bits)')
parser.add_argument('--split', action='store_true',
                    help='split the network into to subnetworks'
                         ' (flows, residuals)')

# video
parser.add_argument('--video', type=str, default='./training/final/alley_1',
                    help='video path (for images use folder path, '
                         'for a video use video path)')
parser.add_argument('--video_scale', type=float, default=1,
                    help='the height and width of a output video will be '
                         'multiplied by video_scale')
parser.add_argument('--n_frames', type=int, default=15,
                    help='the number of frames')
parser.add_argument('--max_frames', type=int, default=None)
parser.add_argument('--quality', type=int, default=20)
parser.add_argument('--codec', type=str, default='h264',
                    choices=['jpeg', 'avif', 'h264'])

# training
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--epochs', type=int, default=5000,
                    help='the number of training epochs')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--grad_clip', type=float, default=0.1)

parser.add_argument('--eval_interval', type=int, default=None)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--visualize', action='store_true')


class Concat(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, inputs):
        return torch.cat([net(inputs) for net in self.nets], -1)


class EmptyContext:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def get_cos_warmup_scheduler(optimizer, total_epoch, warmup_epoch):
    def lr_lambda(epoch):
        if epoch < warmup_epoch:
            return (epoch + 1) / (warmup_epoch + 1)
        return (1 + np.cos(np.math.pi * (epoch - warmup_epoch)
                           / (total_epoch - warmup_epoch))) / 2
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(args, target_frames, keyframes, kf_size, kf_idx, metrics,
         save_path, name='', verbose=False, visualize=True):
    keyframes = keyframes.cuda()
    target_frames = target_frames.cuda()

    # grids
    T, _, H, W = target_frames.size()
    input_grid = make_input_grid(T, H, W).cuda()
    flow_grid = make_flow_grid(H, W).unsqueeze(0).cuda()

    """ PREPARING A NETWORK """
    if args.backbone == 'nerf':
        encoding = PosEncoding(3, np.floor(np.log2([T, H, W])).astype('int'),
                               True, trainable=False)
        x = Symbol('x')
        eq = args.hidden_layers * x**2 \
           + x*(10 + encoding.get_output_size() + args.hidden_layers) \
           - args.ratio * kf_size / ((4 - 2*args.use_amp) * (1 + args.split))
        width = int(np.round(float(max(solve(eq)))/2)*2)

        if not args.split:
            net = NeuralFieldsNetwork(3, 8, width, args.hidden_layers,
                                      encoding, lambda : Swish(bias=True),
                                      reuse=args.reuse)
        else:
            net = Concat(
                [NeuralFieldsNetwork(3, 4, width, args.hidden_layers,
                                     encoding, lambda : Swish(bias=True),
                                     reuse=args.reuse)
                 for i in range(2)])
    elif args.backbone == 'siren':
        x = Symbol('x')
        eq = 3*x**2 + 12*x - args.ratio*kf_size/2
        width = int(np.round(float(max(solve(eq)))/2)*2)

        net = Siren(in_features=3, hidden_features=width, hidden_layers=3,
                    out_features=8, outermost_linear=True)
    else:
        raise ValueError(f'invalid backbone: {args.backbone}')
    net = nn.DataParallel(net.cuda())

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = get_cos_warmup_scheduler(optimizer, args.epochs,
                                         int(0.2*args.epochs))
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model_size = sum(p.numel() for p in net.parameters()) * (4-2*args.use_amp)
    total_size = model_size + keyframe_size

    """ MISC """
    n_metrics = len(metrics)

    header = ', '.join([str(m) for m in metrics] \
                       + ['MSE(flows)', 'MSE(residuals)'])
    perf_logs = []
    default_perfs = np.array(
        [m(keyframes[1:2], target_frames[kf_idx].unsqueeze(0)).cpu() / T
         for m in metrics])

    if verbose:
        print(f'approx. total bytes ({total_size}) = '
              f'model size ({model_size}({width})) + '
              f'keyframe ({keyframe_size})')
        print('keyframe quality:', default_perfs * T)
        print(net)

    if args.use_amp:
        context = lambda : torch.cuda.amp.autocast(enabled=True)
    else:
        context = EmptyContext

    """ START TRAINING """
    net.train()
    # with tqdm.tqdm(range(args.epochs)) as loop:
    #     for epoch in loop:
    with EmptyContext():
        for epoch in range(args.epochs):
            optimizer.zero_grad()

            for i in torch.randperm(T-1).repeat(2).cuda() \
                                        .split(args.batch_size):
                # [B, C, H, W]
                dst = i + (i >= kf_idx)
                targets = torch.index_select(target_frames, 0, dst)
                sources0 = torch.index_select(keyframes, 0, 2*(i >= kf_idx))
                sources1 = torch.broadcast_to(keyframes[[1]],
                                              [len(dst), *keyframes.shape[1:]])

                with context():
                    # [B, H, W, 3]
                    i_grid = torch.index_select(input_grid, 0, dst)
                    f_grid = flow_grid # [1, H, W, 2]

                    # for efficient training
                    # 1. Y
                    size = i_grid.shape[1]
                    indices = torch.randperm(size)[:size//8].sort()[0].cuda()
                    i_grid = torch.index_select(i_grid, 1, indices)
                    f_grid = torch.index_select(f_grid, 1, indices)
                    targets = torch.index_select(targets, -2, indices)

                    # 3. X
                    size = i_grid.shape[2]
                    indices = torch.randperm(size)[:size//8].sort()[0].cuda()
                    i_grid = torch.index_select(i_grid, 2, indices)
                    f_grid = torch.index_select(f_grid, 2, indices)
                    targets = torch.index_select(targets, -1, indices)

                    outputs = net(i_grid) # [B, H, W, 3] -> [B, H, W, 6]

                    flows0, flows1, alpha, res = outputs.split([2, 2, 1, 3],
                                                               -1)
                    alpha = torch.sigmoid(alpha.permute(0, 3, 1, 2))
                    res = torch.tanh(res.permute(0, 3, 1, 2))

                    outputs = warp_frames(sources0, flows0, f_grid) * alpha \
                            + warp_frames(sources1, flows1, f_grid) * (1-alpha)
                    outputs = (outputs + res).clamp(0, 1)

                    size = outputs.size(0)

                    loss = F.mse_loss(outputs, targets) * size/args.batch_size

                    assert not torch.isnan(loss)

                    if args.use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # update
                if args.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                   args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            scheduler.step()

            if (epoch + 1) % args.eval_interval == 0: # eval
                perf_logs.append(np.zeros(n_metrics))
                perf_logs[-1][:n_metrics] = default_perfs

                net.eval()
                total_time = 0

                for i in range(T-1):
                    dst = i + (i >= kf_idx)
                    sources0 = keyframes[[2*(i >= kf_idx)]]
                    sources1 = keyframes[[1]]

                    with torch.no_grad():
                        with context():
                            start = time.time()
                            # [B, H, W, 3] -> [B, H, W, 8]
                            outputs = net(input_grid[[dst]])

                            flows0, flows1, alpha, res = outputs.split(
                                [2, 2, 1, 3], -1)
                            alpha = torch.sigmoid(alpha.permute(0, 3, 1, 2))
                            res = torch.tanh(res.permute(0, 3, 1, 2))

                            outputs = (
                                warp_frames(sources0, flows0, flow_grid)*alpha
                              + warp_frames(sources1, flows1, flow_grid)
                                * (1-alpha)
                              + res).clamp(0, 1)
                            total_time += time.time() - start

                    if visualize:
                        postfix = f'{name}_{epoch:04d}:{dst}.jpg'
                        save_image(
                            warped0,
                            os.path.join(save_path, f'warped0_{postfix}'))
                        save_image(
                            warped1,
                            os.path.join(save_path, f'warped1_{postfix}'))
                        save_image(alpha[0],
                                   os.path.join(save_path, f'alpha_{postfix}'))
                        save_image(mu,
                                   os.path.join(save_path, f'mu_{postfix}'))
                        save_image(outputs[0, -3:],
                                   os.path.join(save_path, f'res_{postfix}'))
                        save_image(reconstructed_frame[0],
                                   os.path.join(save_path, f'final_{postfix}'))
                        save_image(target_frames[dst],
                                   os.path.join(save_path,
                                                f'target_{name}:{dst}.jpg'))

                    for j in range(n_metrics):
                        value = metrics[j](
                            outputs.float(), target_frames[[dst]]).item()
                        perf_logs[-1][j] += value / T

                net.train()

                # print(f'FPS: {T / total_time}')

                postfix = {str(metrics[i]): perf_logs[-1][i]
                           for i in range(n_metrics)} # test performance
                print(postfix) # loop.set_postfix(postfix)

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

    return perf_logs[-1], model_size


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
    keyframe_indices = ((gop_indices[:-1] + gop_indices[1:])/2).astype('int')

    # keyframes
    keyframes = []
    keyframe_sizes = []

    for i, kf_idx in enumerate(keyframe_indices):
        keyframe, keyframe_size = save_keyframe(
            target_frames[kf_idx], args.quality,
            os.path.join(save_path, f'keyframe{i:02d}.jpeg'),
            codec=args.codec)
        keyframes.append(keyframe)
        keyframe_sizes.append(keyframe_size)

    metrics = [PSNR(), SSIM()]

    # compress
    performances = []
    model_sizes = []
    start = time.time()

    for i in range(n_gop): 
        # print(gop_indices[i:i+2], keyframe_indices[i])
        perfs, size = main(
            args,
            target_frames[gop_indices[i]:gop_indices[i+1]],
            torch.stack([keyframes[max(0, i-1)],
                         keyframes[i],
                         keyframes[min(n_gop-1, i+1)]], 0),
            keyframe_sizes[i],
            keyframe_indices[i]-gop_indices[i], metrics,
            save_path, f'{i}', args.verbose, args.visualize)
        performances.append(perfs)
        model_sizes.append(size)

    total_keyframe_size = sum(keyframe_sizes)
    total_model_size = sum(model_sizes)
    total_size = total_keyframe_size + total_model_size

    performances = np.stack(performances, 0)
    weights = ((gop_indices[1:] - gop_indices[:-1]) / total_frames)[..., None]
    performances = np.sum(performances * weights, 0)

    print(f'total performances: {performances} ({time.time() - start} sec)')
    print(f'total_size: {total_size} '
          f'(kf: {total_keyframe_size}, mdl: {total_model_size})')
    print(f'bpp: {8*total_size/(total_frames * target_frames.shape[-2] * target_frames.shape[-1])}')

