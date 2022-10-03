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


def load_video(path, *args, **kwargs):
    if os.path.isdir(path):
        return load_video_from_images(path, *args, **kwargs)
    elif os.path.splitext(path)[1].lower() in ['.avi', '.mp4']:
        return load_video_from_video(path, *args, **kwargs)
    else:
        raise ValueError('Path must be one of a directory containing '
                         'PNG files or a ".avi" formatted video')


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


def load_video_from_video(video_path, start_frame=None, n_frames=None,
                          scale=1., antialias=False):
    name, ext = os.path.splitext(video_path)

    # assert ext.lower() == '.avi', 'currently only AVI format is supported'

    frames = skvideo.io.vread(f'{name}{ext}')
    if start_frame is None:
        start_frame = 0
    if n_frames is None:
        n_frames = len(frames)
    frames = frames[start_frame:start_frame+n_frames]
    frames = torch.from_numpy(frames).float() / 255. # [0, 255] to [0, 1]
    frames = frames.permute(0, 3, 1, 2) # to [time, chan, height, width]

    if scale != 1:
        frames = torchvision.transforms.functional.resize(
            frames,
            tuple(int(s*scale) for s in frames.size()[-2:]), # new video size
            antialias=antialias)

    return frames


def extract_flows(frames):
    '''
    extracts backward flows
    I_{t+1}(x, y) = I_t(x+u, y+v)
    '''
    if frames.size(0) <= 1: # single frame
        return torch.zeros(0, *frames.shape[-2:], 2).cuda()
    model = load_optical_flow_estimator()
    padder = InputPadder(frames.shape)

    frames = frames.cuda()
    image1, image2 = padder.pad(frames[1:], frames[:-1])
    _, flows = model(image1*255., image2*255., iters=12, test_mode=True)
    flows = padder.unpad(flows).detach()

    return flows.permute(0, 2, 3, 1) # to (N, H, W, C)


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


def save_keyframe(keyframe, quality, save_path, codec='jpeg'):
    '''
    inputs: keyframe (torch.Tensor or torch.cuda.Tensor)
            quality (int)
            save_path (str)
    outputs: keyframe (encoded and decoded frame)
             keyframe_size (int)
    '''
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, 'keyframe.jpeg')

    keyframe = (keyframe * 255).permute(1, 2, 0)
    keyframe = keyframe.numpy().astype('uint8')
    keyframe = PIL.Image.fromarray(keyframe)

    if codec == 'jpeg':
        save_path = f'{os.path.splitext(save_path)[0]}.jpeg'
        keyframe.save(save_path, quality=quality, subsampling=0)
    elif codec == 'avif':
        import pillow_avif
        save_path = f'{os.path.splitext(save_path)[0]}.png'
        keyframe.save(save_path)
        os.system(f'avifenc --min 0 --max 63 -a end-usage=q -a '
                  f'cq-level={quality} -a tune=ssim {save_path} '
                  f'{save_path.replace(".png", ".avif")}')
        os.remove(save_path)
        save_path = save_path.replace('.png', '.avif')
    elif codec == 'h264':
        save_path = f'{os.path.splitext(save_path)[0]}.png'
        keyframe.save(save_path)
        new_path = save_path.replace('.png', '.mp4')
        if os.path.exists(new_path):
            os.remove(new_path)

        os.system(f'ffmpeg -hide_banner -loglevel error '
                  f'-i {save_path} -vcodec libx264 '
                  f'-crf {quality} {new_path}')
        os.remove(save_path)
        save_path = new_path
    else:
        raise ValueError(f'unsupported codec: {codec}')
    keyframe_size = os.stat(save_path).st_size

    if codec not in ['h264']:
        keyframe = transforms.functional.to_tensor(
            PIL.Image.open(save_path))
    else:
        keyframe = load_video(save_path)[0]

    return keyframe, keyframe_size


def make_input_grid(*input_size, minvalue=-1, maxvalue=1):
    # generate grid for a given input size
    # 2D coordinates: use make_input_grid(H, W) -> (H, W, 2) shaped tensor
    # 3D coordinates: use make_input_grid(T, H, W) -> (T, H, W, 3) shaped tensor
    return torch.stack(
        torch.meshgrid(*[torch.linspace(minvalue, maxvalue, int(S))
                         for S in input_size]), -1)


# (x,y)
def make_flow_grid(H, W):
    # generates (H, W, 2) shaped tensor
    flow_grid = torch.stack(torch.meshgrid(torch.arange(0, H),
                                           torch.arange(0, W)), -1).float()
    return torch.flip(flow_grid, (-1,)) # from (y, x) to (x, y)


def warp_frames(source_frames, flows, flow_grid):
    if source_frames.ndim == 3:
        source_frames = source_frames.unsqueeze(0)
    if flows.ndim == 3:
        flows = flows.unsqueeze(0)

    return F.grid_sample(source_frames,
                         apply_flow(flow_grid, flows, *source_frames.shape[-2:]),
                         'bicubic',
                         padding_mode='reflection', align_corners=True)


def apply_flow(prev_coords, flow, H=None, W=None, normalize=True):
    # assume flow_grid and pred_flow are pixel locations
    next_coords = prev_coords.to(flow.device) + flow
    if normalize:
        # normalize to [-1, 1]
        assert H is not None and W is not None, 'both H,W must not be None'
        next_coords = 2 * next_coords \
                    / torch.tensor([[[[W-1, H-1]]]]).to(next_coords.device) - 1 
    return next_coords

