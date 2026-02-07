#!/bin/bash

# 运行训练脚本并将输出重定向到日志文件
CUDA_VISIBLE_DEVICES=0 nohup python train.py > 0_ablation_normal_attention.log 2>&1 &

