#!/usr/bin/env bash

CONFIG_FILE=${1:-"./configs/default.yaml"}

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    src/train.py \
    --config ${CONFIG_FILE} \
    --ddp
