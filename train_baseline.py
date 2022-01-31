import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm

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
parser.add_argument('--hidden_features', type=int, default=96,
                    help='the number of hidden units per layer')
parser.add_argument('--hidden_layers', type=int, default=3,
                    help='the number of layers (default: 3)')
parser.add_argument('--use_amp', action='store_true',
                    help='use automatic mixed precision (32 to 16 bits)')

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
parser.add_argument('--batch_size', type=int, default=None)

parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--save_logs_interval', type=int, default=10000)


def apply_weight_decay(net, weight_decay):
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if 'weight' in name or 'bias' in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]


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
    input_grid = make_input_grid(T, H, W, minvalue=0, maxvalue=1)

    """ PREPARING A NETWORK """
    '''
    net = Siren(in_features=3,
                hidden_features=args.hidden_features,
                hidden_layers=args.hidden_layers,
                out_features=3,
                outermost_linear=True)
    '''
    net = NeuralFieldsNetwork(3, 3, args.hidden_features, args.hidden_layers,
                              MultiHashEncoding(target_frames, 6, 16, 1),
                              Swish) # 'ReLU')
    # net = nn.DataParallel(net.cuda())
    net = net.cuda()

    optimizer = optim.Adam(apply_weight_decay(net, 1e-6), betas=(0.9, 0.99),
                           eps=1e-15, lr=args.lr)
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        model_size = sum(p.numel() for p in net.parameters()) * 2
    else:
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

    n_samples = T * H * W
    batch_size = H * W if args.batch_size is None else args.batch_size
    n_steps = n_samples // batch_size
    indices_scaler = torch.tensor([T, H, W]) - 1

    """ START TRAINING """
    net.train()
    with tqdm.tqdm(range(args.epochs)) as loop:
        for epoch in loop:
            optimizer.zero_grad()

            is_eval_epoch = (epoch + 1) % args.eval_interval == 0

            if is_eval_epoch:
                perf_logs.append(np.zeros(n_metrics))

            for i in range(n_steps):
                indices = torch.round(torch.rand(batch_size, 3)
                                      * indices_scaler).long()
                inputs = input_grid[indices[..., 0], indices[..., 1],
                                    indices[..., 2]]
                targets = target_frames[indices[..., 0], :,
                                        indices[..., 1], indices[..., 2]]

                if args.use_amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = torch.sigmoid(net(inputs))
                        loss = F.mse_loss(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = torch.sigmoid(net(inputs))
                    loss = F.mse_loss(outputs, targets)
                    loss.backward()
                    optimizer.step()

            if is_eval_epoch:
                net.eval()

                for i in range(T):
                    inputs = input_grid[i]
                    if args.use_amp:
                        with torch.cuda.amp.autocast(enabled=True):
                            outputs = torch.sigmoid(net(inputs))
                    else:
                        outputs = torch.sigmoid(net(inputs))
                    outputs = outputs.permute(2, 0, 1).unsqueeze(0) # RGB

                    for j in range(n_metrics):
                        perf_logs[-1][j] += metrics[j](
                            outputs.float(),
                            target_frames[i].unsqueeze(0)).item() / T

                net.train()

                if perf_logs[-1][0] < 20:
                    import pdb; pdb.set_trace()

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

