#!/usr/bin/env bash

CONFIG_FILE=${1:-"./config/config.yaml"}

torchrun --nproc_per_node=7 \
    src/train.py \
    --config ${CONFIG_FILE} \
    --ddp
