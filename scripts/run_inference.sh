# scripts/run_inference.sh
#!/usr/bin/env bash

CONFIG_FILE=${1:-"./config/config.yaml"}

python src/main.py --config ${CONFIG_FILE} --mode eval
