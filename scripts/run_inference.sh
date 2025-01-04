#!/usr/bin/env bash

CONFIG_FILE=${1:-"./configs/default.yaml"}

python src/main.py --config ${CONFIG_FILE} --mode eval
