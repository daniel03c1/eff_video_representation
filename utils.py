import PIL
import matplotlib.pyplot as plt
import numpy as np
import os.path
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8

        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2,
                         pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2,
                         0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def load_optical_flow_estimator(checkpoint='GMA/checkpoints/gma-sintel.pth'):
    class Args:
        model = checkpoint
        model_name = 'GMA'
        num_heads = 1
        position_only = False
        position_and_content = False
        mixed_precision = False

    from GMA.utils import RAFTGMA

    m_args = Args()
    model = RAFTGMA(m_args)
    state_dict = torch.load(m_args.model)
    keys = state_dict.keys()
    new_state_dict = {}
    for key in keys:
        new_state_dict[key[7:]] = state_dict[key]
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def load_video(data_dir, video_name, T, start_frame, key_frame_quality, tag, use_estimator=False):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((218, 512)), # (1024, 436)
    ])
    video_dir = data_dir + "/final/{}/".format(video_name)
    flow_dir = data_dir + "/flow/{}/".format(video_name)

    target_frames = []
    target_flow = []

    # frame_0001 is the first frame, so t+1
    for t in range(start_frame, start_frame+T):
        target_frames.append(tf(PIL.Image.open(video_dir+"frame_{:04d}.png".format(t+1))))
    # the last one more frame
    target_frames.append(tf(PIL.Image.open(video_dir+"frame_{:04d}.png".format(t+2))))
    target_frames = torch.stack(target_frames, 0)

    if not use_estimator:
        for t in range(start_frame, start_frame+T):
            target_flow.append(tf(cv2.readOpticalFlow(flow_dir+"frame_{:04d}.flo".format(t+1)))/2)
        target_flow = torch.stack(target_flow, 0)
    else:
        model = load_optical_flow_estimator()
        target_frames = target_frames.cuda()
        padder = InputPadder(target_frames.shape)

        image1, image2 = padder.pad(target_frames[:T], target_frames[1:T+1])
        _, target_flow = model(image1*255, image2*255, iters=12, test_mode=True)
        target_flow = padder.unpad(target_flow).detach()
        del model, padder
    
    # load key frame as jpeg
    key_index = start_frame + int((T/2))
    dirname = "./results/{}/{}/".format(video_name, tag)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fname = dirname+"key_frame_{:05d}.jpg".format(key_index)
    key_frame = PIL.Image.open(video_dir+"frame_{:04d}.png".format(key_index+1))
    key_frame = key_frame.resize((512,218))
    key_frame.save(fname, quality=key_frame_quality, subsampling=0)
    key_frame_size = os.stat(fname).st_size
    to_tensor = transforms.ToTensor()
    key_frame = to_tensor(PIL.Image.open(fname))

    return target_frames, target_flow, key_frame, key_frame_size


def show_tensor_to_image(tensor, file_name):
    torchvision.utils.save_image(tensor, './{}.jpg'.format(file_name))


# (t,x,y)
def make_input_grid(T, H, W):
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    t = torch.linspace(-1, 1, T)
    input_grid = torch.stack(torch.meshgrid(t,y,x),-1)
    input_grid = torch.stack((input_grid[:,:,:,0],input_grid[:,:,:,2], input_grid[:,:,:,1]),-1)
    return input_grid


# (x,y)
def make_flow_grid(H, W):
    # generates (H, W, 2) shaped tensor
    flow_grid = torch.stack(torch.meshgrid(torch.arange(0, H),
                                           torch.arange(0, W)), -1).float()
    return torch.flip(flow_grid, (-1,)) # from (y, x) to (x, y)


def apply_flow(flow_grid, pred_flow, H, W, direction='rl'):
    # making flow grid for grid_sample
    # (right -> left) or (left -> right)
    if direction=='rl':
        flow_grid_shift = flow_grid + pred_flow
    else:
        flow_grid_shift = flow_grid - pred_flow
    flow_grid_shift_x = 2.0 * flow_grid_shift[:, :, :, 0] / (W - 1) - 1.0
    flow_grid_shift_y = 2.0 * flow_grid_shift[:, :, :, 1] / (H - 1) - 1.0
    flow_grid_shift = torch.stack((flow_grid_shift_x, flow_grid_shift_y), -1)
    return flow_grid_shift

