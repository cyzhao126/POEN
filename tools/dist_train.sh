#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_gpus.py $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}

# sh C:/zcy/Video-Swin-Transformer-master/Video-Swin-Transformer-master/tools/dist_train.sh C:/zcy/Video-Swin-Transformer-master/Video-Swin-Transformer-master/configs/recognition/swin/multiview_track1.py 2 --cfg-options load_from='C:/zcy/Video-Swin-Transformer-master/Video-Swin-Transformer-master/data/TRACK1/swin_base_patch244_window877_kinetics400_22k.pth'