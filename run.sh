#!/bin/bash

output_dir=models-woz/exp
cuda=0
target_slot='all'
bert_dir='/Users/kapilchandra/.pytorch_pretrained_bert'

# Running multiWOZ
CUDA_VISIBLE_DEVICES=$cuda python3 data_prep.py
