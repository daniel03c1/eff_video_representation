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
parser.add_argument('--keyframe_loc', type=str, default='mid',
                    choices=['front', 'mid', 'last'])
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--n_frames', type=int, default=7,
                    help='the number of frames')
parser.add_argument('--jpeg-quality', type=int, default=98)

# training
parser.add_argument('--seed', type=int, default=50236, metavar='S',
                    help='random seed (default: 50236)')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--epochs', type=int, default=30000,
                    help='the number of training epochs')
parser.add_argument('--flow-warmup-step', type=int, default=2000,
                    help='flow only training warmup (default: 2000)')
parser.add_argument('--image-warmup-step', type=int, default=5000,
                    help='flow only training warmup (default: 5000)')

parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--save_logs_interval', type=int, default=10000)


def get_keyframe_idx(keyframe_loc: str, n_frames: int):
    if keyframe_loc == 'front':
        return 0
    elif keyframe_loc == 'mid':
        return n_frames // 2
    elif keyframe_loc == 'last':
        return n_frames - 1
    else:
        raise ValueError(f'invalid keyframe_loc ({keyframe_loc})')


def get_target_flows_residuals(frames, keyframe, keyframe_idx, flow_grid):
    def stitch(frames, keyframe_in_front):
        # stitch keyframe onto frames
        if frames.size(0) == 0:
            return frames

        if keyframe_in_front:
            return torch.cat([keyframe, frames], 0)
        else:
            return torch.cat([frames, keyframe], 0)

    if keyframe.ndim == 3:
        keyframe = keyframe.unsqueeze(0)

    target_flows = torch.cat(
        [extract_flows(torch.flip(stitch(target_frames[:kf_idx], False), (0,))),
         extract_flows(stitch(target_frames[kf_idx+1:], True))],
        0)

    warped = torch.cat(
        [warp_frames(stitch(target_frames[1:kf_idx], False),
                     target_flows[:kf_idx], flow_grid),
         warp_frames(stitch(target_frames[kf_idx+1:-1], True),
                     target_flows[kf_idx:], flow_grid)],
        0)
    target_residuals = torch.cat(
        [target_frames[:kf_idx], target_frames[kf_idx+1:]], 0) - warped

    return target_flows, target_residuals


def visualize():
    raise NotImplementedError()


if __name__=='__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    print('configs')
    print(str(vars(args)))

    """ PREPARING A VIDEO """
    target_frames = load_video_from_images(
        args.video_path, args.start_frame, args.n_frames, args.video_scale)

    # keyframe
    kf_idx = get_keyframe_idx(args.keyframe_loc, len(target_frames))
    keyframe, keyframe_size = save_keyframe(
        target_frames[kf_idx], args.jpeg_quality, '000.jpeg')
    keyframe = keyframe.cuda()
    target_frames = target_frames.cuda()

    # grids
    T, _, H, W = target_frames.size()
    input_grid = make_input_grid(T, H, W)
    flow_grid = make_flow_grid(H, W).unsqueeze(0)

    # flows
    target_flows, target_residuals = get_target_flows_residuals(
        target_frames, keyframe, kf_idx, flow_grid)

    """ PREPARING A NETWORK """
    net = Siren(in_features=3,
                hidden_features=args.hidden_features,
                hidden_layers=args.hidden_layers,
                out_features=5,
                outermost_linear=True)
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    model_size = sum(p.numel() for p in net.parameters()) * 4
    print(f'total bytes ({model_size+keyframe_size}) = '
          f'model size ({model_size}) + keyframe ({keyframe_size})')

    """ MISC """
    metrics = [PSNR(), SSIM()]
    n_metrics = len(metrics)

    header = ', '.join([str(m) for m in metrics] \
                       + ['MSE(flows)', 'MSE(residuals)'])
    perf_logs = []
    default_perfs = np.array(
        [m(keyframe.unsqueeze(0), target_frames[kf_idx].unsqueeze(0)) / T
         for m in metrics])

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
                perf_logs.append(np.zeros(n_metrics + 2))
                perf_logs[-1][:n_metrics] = default_perfs

            for i in range(T-1):
                backward = i < kf_idx
                if backward:
                    src = kf_idx - i
                    dst = src - 1
                else:
                    src = i
                    dst = src + 1

                outputs = net(input_grid[dst])
                flows, outputs = outputs[..., :2], outputs[..., 2:]
                outputs = outputs.permute(2, 0, 1) # residuals

                if i == 0 or i == kf_idx:
                    src_frame = keyframe
                else:
                    if epoch <= args.image_warmup_step:
                        src_frame = target_frames[src]
                    else:
                        src_frame = reconstructed_frame.detach()
                reconstructed_frame = warp_frames(src_frame, flows, flow_grid) \
                                    + outputs.unsqueeze(0)

                # flows and residuals losses
                flows_loss = F.mse_loss(
                    flows, target_flows[dst - (not backward)])
                residuals_loss = F.mse_loss(
                    outputs, target_residuals[dst - (not backward)])

                if epoch <= args.flow_warmup_step:
                    loss = flows_loss + residuals_loss
                else:
                    loss = F.mse_loss(reconstructed_frame, target_frames[i])
                loss.backward()

                if not is_eval_epoch:
                    continue

                # evaluate
                perf_logs[-1][-2] += flows_loss.item() / T
                perf_logs[-1][-1] += residuals_loss.item() / T

            # update per epoch
            optimizer.step()

            if is_eval_epoch:
                net.eval()

                for i in range(T-1):
                    backward = i < kf_idx
                    if backward:
                        src = kf_idx - i
                        dst = src - 1
                    else:
                        src = i
                        dst = src + 1

                    outputs = net(input_grid[dst])
                    flows, outputs = outputs[..., :2], outputs[..., 2:]
                    outputs = outputs.permute(2, 0, 1).unsqueeze(0) # residuals

                    if i == 0 or i == kf_idx:
                        src_frame = keyframe
                    else:
                        src_frame = reconstructed_frame.detach()
                    reconstructed_frame = warp_frames(
                        src_frame, flows, flow_grid) + outputs

                    for i in range(n_metrics):
                        perf_logs[-1][i] += metrics[i](
                            reconstructed_frame,
                            target_frames[dst].unsqueeze(0)).item() / T

                postfix = {str(metrics[i]): perf_logs[-1][i]
                           for i in range(n_metrics)} # test performance
                postfix['MSE(flows)'] = perf_logs[-1][-2]
                postfix['MSE(residuals)'] = perf_logs[-1][-2]
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

