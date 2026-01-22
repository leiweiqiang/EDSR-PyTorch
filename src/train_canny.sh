#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python main.py --scale 8 --patch_size 512 --save edsr_bicubic_edge_x8 --reset --split_batch 4 --dir_data /mnt/T5_4T/workspace/dataset --data_range 1-800/801-810

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --n_GPUs 4 --scale 8 --patch_size 512 --load edsr_bicubic_edge_x8 --resume -1 --split_batch 4 --dir_data /mnt/T5_4T/workspace/dataset --data_range 1-800/801-810
# python main.py --scale 8 --patch_size 512 --load edsr_bicubic_edge_x8 --resume -1 --split_batch 4 --dir_data /mnt/T5_4T/workspace/dataset --data_range 1-800/801-810
