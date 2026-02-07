#!/bin/bash

# Run the training script and redirect output to a log file
CUDA_VISIBLE_DEVICES=0 nohup python train.py > 0_ablation_normal_attention.log 2>&1 &

