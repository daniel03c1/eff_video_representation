# Neural Residual Flow Fields for Efficient Video Representations ([arxiv](https://arxiv.org/abs/2201.04329))
Accepted for ACCV 2022.

You can compress or convert a video using this repository.
As an input, our codes only accept an image folder, ".avi" or ".mp4" encoded video file.

There are two main files.
One is "encode_baseline.py," which compresses a video using color-based INR.
Another is "encode.py," which compresses a video using NRFF and multiple reference frames.

<img width="1284" alt="multi_main" src="https://user-images.githubusercontent.com/48237485/193852294-8f77e958-5553-415f-9ac6-8689e4629185.png">

### 1. Preparation
You can download video files from the following urls: [MPI SINTEL](http://sintel.is.tue.mpg.de/), [UVG](http://ultravideo.fi/#testsequences).

Since we do not support ".yuv" videos, we strongly recommend you convert ".yuv" videos using lossless video compression, such as x264 with a crf of 0.

### 2. Video Compression
1. Common options

    "--tag": The results will be stored in a folder named "tag".
    
    "--use_amp": (recommended) reduce the bit precision from 32 to 16.

    "--video": the path of the input video. This can be an image folder or a video file.

    "--n_frames": the size of a group of pictures (GOP). The input video will be automatically separated so that the size of each GOP will be similar to the predefined "n_frames".

    "--lr": the initial learning rate.

    "--epochs": the number of epochs.
    
    "--batch_size": batch size.
    
    "--max_frames": the maximum number of frames to encode in the given video. If it is not set, it will automatically encode all the video frames, and if it is set, it will only encode the first "max_frames" frames.
   

   
2. Other options ("encode_baseline.py")

    "--bpp": It controls bits per pixel (bpp). The network size will be automatically set to match a predefined bpp.


3. Other options ("encode.py")

    "--ratio": the ratio of the network size to the key frame size. If it is set to 1/4, the network size will be automatically set to one quarter of the key frame size.

    "--quality": the quality factor for key frame compression. For example, if "codec" is "jpeg" and "quality" is set to 85, it will compress the key frame with "jpeg" with a quality factor of 85. Likewise, if "codec" is "h264" and "quality" is set to 15, it will compress the key frame with "h264" with a crf of 15.
    
    "--split": whether to separate the network into two sub-networks, one of which is for optical flows and the other is for residuals. If the option is not set, the network will not be separated.
    
    "--codec": key frame compression codec. It currently supports only "jpeg", "avif", and "h264".


### 3. Examples

```bash
# Compressing "alley_1" video from SINTEL using SIREN.
python3 encode_baseline.py --use_amp --n_frames=5 --bpp=0.2 --tag=SIREN_ALLEY_1 --video=training/final/alley_1 --lr=1e-5

# Compressing "ReadySteadyGo" video from UVG using multiple reference NRFF with split option.
python3 encode.py --use_amp --n_frames=15 --ratio=1 --tag=mNRFF_RSG --video=ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.mp4 --lr=1e-3 --split
```


### 4. Examples
https://user-images.githubusercontent.com/48237485/148535439-0f24fadd-07a9-4a99-af28-59c3b67ee2c1.mp4
https://user-images.githubusercontent.com/48237485/148956883-8050fbbc-bfca-4222-8dff-21a6f7492d13.mp4
