# Neural Residual Flow Fields for Efficient Video Representations

### 1. Download MPI sintel dataset 
Download MPI sintel dataset from [here](http://sintel.is.tue.mpg.de/)

### 2. GMA optical flow estimator
To obtain optical flow estimations for pretraining, we are using GMA from [here](https://github.com/zacjiang/GMA). Note that it dose not have to do with our identity.

### 3. Training 
Training neural residual flow fields (NRFF)
```bash
# frame 0 - 6
python train_video_flow_midkey.py --use-estimator --lr 0.0005 --training-step 30000 --data-dir {sintel dataset training directory} --video-name alley_1 --start-frame 0 --num-frames 7 --jpeg-quality 98 --hidden-features 96 --use-estimator --tag start0_jq98_hf96
# frame 7 - 13
python train_video_flow_midkey.py --use-estimator --lr 0.0005 --training-step 30000 --data-dir {sintel dataset training directory} --video-name alley_1 --start-frame 7 --num-frames 7 --jpeg-quality 98 --hidden-features 96 --use-estimator --tag start7_jq98_hf96
# frame 14 - 20
python train_video_flow_midkey.py --use-estimator --lr 0.0005 --training-step 30000 --data-dir {sintel dataset training directory} --video-name alley_1 --start-frame 14 --num-frames 7 --jpeg-quality 98 --hidden-features 96 --use-estimator --tag start14_jq98_hf96
# frame 21 - 27
python train_video_flow_midkey.py --use-estimator --lr 0.0005 --training-step 30000 --data-dir {sintel dataset training directory} --video-name alley_1 --start-frame 21 --num-frames 7 --jpeg-quality 98 --hidden-features 96 --use-estimator --tag start21_jq98_hf96
```

Training baseline (SIREN)
```bash
python train_video.py --data-dir {sintel dataset training directory} --video-name alley_1 --hidden-features 256 --num-frames 28 --lr 0.001 --training-step 30000 --tag baseline_siren_hf256
```

