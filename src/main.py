# src/main.py

import argparse
import yaml
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--mode", type=str, default="train", help="train or eval")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        from .train import main_train
        main_train(config)
    elif args.mode == "eval":
        print("[안내] 평가 모드는 아직 구현되지 않았습니다. 별도의 eval 스크립트를 만들어 사용하세요.")
    else:
        print(f"[에러] 지원되지 않는 모드: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
