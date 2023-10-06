# Anomaly-GAT-BERT: Transformer-based Anomaly Detector

This is the code for **Self-supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme**.
This code is forked from: [AnomalyBERT](https://github.com/Jhryu30/AnomalyBERT/tree/main)

The architecture is inspired by [AnomalyBERT](https://arxiv.org/abs/2305.04468v1) and [MTAD-GAT](https://arxiv.org/pdf/2009.02040.pdf)

## Installation

Please clone the repository at `path/to/repository/` and install the packages in `requirements.txt`.
It is recommend installing Python 3.8 and Pytorch 1.9 with CUDA.

```
git clone https://github.com/MRTCc/Anomaly-GAT-BERT.git

conda create --name your_env_name python=3.8
conda activate your_env_name

pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html  # example CUDA setting
pip install -r requirements.txt
```

We use five public datasets, SMAP, MSL, SMD, SWaT, and WADI.
Following the instruction in [here](utils/DATA_PREPARATION.md), you can download and preprocess the datasets.
After preprocessing, you need to edit your dataset directory in `utils/config.py`.

```
DATASET_DIR = 'path/to/dataset/processed/'
```


## Training

To train with deafault options:
```
python3 train.py --dataset <dataset>
```

Default training parameters can be found in `train.py`:
```
--gpu_id=0
--lr=0.0001
--max_steps=150000
--summary_steps=500
--checkpoint=None
--initial_iter=0
--dataset=SMAP
--replacing_data=None
--batch_size=16
--n_features=512
--patch_size=4
--d_embed=512
--n_layer=6
--dropout=0.1
--replacing_rate_max=0.15
--soft_replacing=0.5
--uniform_replacing=0.15
--peak_noising=0.15
--length_adjusting=0.0
--white_noising=0.0
--flip_replacing_interval=all
--replacing_weight=0.7
--window_sliding=16
--data_division=None
--loss=bce
--total_loss=0.2
--partial_loss=1.0
--contrastive_loss=0.0
--grad_clip_norm=1.0
--default_options=None
--alpha=0.2
```


## Anomaly score estimation and metric computation

To estimate anomaly scores of test data with the trained model, run the `estimate.py` code.
For example, you can estimate anomaly scores of SMAP test set divided by channel with window sliding of 16.

```
python3 estimate.py --dataset=SMAP --model=logs/YYMMDDhhmmss_SMAP/model.pt --state_dict=logs/YYMMDDhhmmss_SMAP/state_dict.pt \
--window_sliding=16
```

Now you will obtain results (npy) file that contains the estimated anomaly scores.
With the results file, you can compute F1-score with and without the point adjustment by running:

```
python3 compute_metrics.py --dataset=SMAP --result=logs/YYMMDDhhmmss_SMAP/state_dict_results.npy
```

If you want to customize the estimation or computation settings, please check the options in `estimate.py` and `compute_metrics.py`.
