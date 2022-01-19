import PIL
import cv2
import glob
import numpy as np
import os
import skvideo.io
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


def load_video_from_images(image_folder, start_frame=None, n_frames=None,
                           scale=1., antialias=False):
    frames = sorted(glob.glob(os.path.join(image_folder, '*.png')))

    if start_frame is None:
        start_frame = 0
    if n_frames is None:
        n_frames = len(frames)

    frames = [transforms.functional.to_tensor(PIL.Image.open(f))
              for f in frames[start_frame:start_frame+n_frames]]
    frames = torch.stack(frames, 0)

    if scale != 1:
        frames = torchvision.transforms.functional.resize(
            frames,
            tuple(int(s*scale) for s in frames.size()[-2:]),
            antialias=antialias)

    return frames


def load_video_from_video():
    raise NotImplementedError()


def extract_flows(frames):
    '''
    extracts backward flows
    I_{t+1}(x, y) = I_t(x+u, y+v)
    '''
    model = load_optical_flow_estimator()
    padder = InputPadder(frames.shape)

    frames = frames.cuda()
    image1, image2 = padder.pad(frames[1:], frames[:-1])
    _, flows = model(image1*255., image2*255., iters=12, test_mode=True)
    flows = padder.unpad(flows).detach()

    return flows


def save_keyframe(keyframe, quality_factor, save_path):
    '''
    inputs: keyframe (torch.Tensor or torch.cuda.Tensor)
            quality_factor (int)
            save_path (str)
    outputs: keyframe (encoded and decoded frame)
             keyframe_size (int)
    '''
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, 'keyframe.jpeg')

    keyframe = (keyframe * 255).permute(1, 2, 0)
    keyframe = keyframe.numpy().astype('uint8')
    keyframe = PIL.Image.fromarray(keyframe)
    keyframe.save(save_path, quality=quality_factor, subsampling=0)
    keyframe_nbyte = os.stat(save_path).st_size
    keyframe = transforms.functional.to_tensor(PIL.Image.open(save_path))
    os.remove(save_path)

    return keyframe, keyframe_nbyte


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

