# Neural Residual Flow Fields for Efficient Video Representations ([arxiv](https://arxiv.org/abs/2201.04329))

### 1. Download MPI sintel dataset 
Download MPI sintel dataset from [here](http://sintel.is.tue.mpg.de/)

### 2. Optical flow estimator
For flows and residuals pretraining, we have used GMA from [here](https://github.com/zacjiang/GMA).
However, using other optical flow estimators is also possible.

### 3. Training examples
* the default video_path is already set to ./training/final/alley_1. If you want to change the video directory, use --video_path.

Training neural residual flow fields (NRFF) for alley_1

```bash
# frame 0 - 6
python train_res_flow.py --start_frame=0 --n_frames=7 --tag=alley_1_start0_jq98_hf96
# frame 7 - 13
python train_res_flow.py --start_frame=7 --n_frames=7 --tag=alley_1_start7_jq98_hf96
# frame 14 - 20
python train_res_flow.py --start_frame=14 --n_frames=7 --tag=alley_1_start14_jq98_hf96
# frame 21 - 27
python train_res_flow.py --start_frame=21 --n_frames=7 --tag=alley_1_start21_jq98_hf96
```

Training baseline (SIREN)
```bash
python train_baseline.py --hidden-features=256 --n_frames=28 --lr=0.001 --tag baseline_siren_hf256
```

### 4. Examples
https://user-images.githubusercontent.com/48237485/148535439-0f24fadd-07a9-4a99-af28-59c3b67ee2c1.mp4

https://user-images.githubusercontent.com/48237485/148956883-8050fbbc-bfca-4222-8dff-21a6f7492d13.mp4
