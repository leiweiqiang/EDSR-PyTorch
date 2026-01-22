import os

from data import common

import numpy as np
import imageio
import cv2

import torch
import torch.utils.data as data

class Demo(data.Dataset):
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            if f.find('.png') >= 0 or f.find('.jp') >= 0:
                self.filelist.append(os.path.join(args.dir_demo, f))
        self.filelist.sort()

    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        lr = imageio.imread(self.filelist[idx])
        lr, = common.set_channel(lr, n_channels=self.args.n_colors)
        
        # Upscale LR to HQ size using bicubic
        scale = self.scale[self.idx_scale]
        lr_upscaled = common.bicubic_upsample(lr, scale)
        
        if self.args.dir_edge:
            edge_path = None
            for ext in ('.png', '.jpg', '.jpeg'):
                candidate = os.path.join(self.args.dir_edge, filename + ext)
                if os.path.isfile(candidate):
                    edge_path = candidate
                    break
            if edge_path is None:
                raise FileNotFoundError(
                    'Edge map not found for {} in {}'.format(
                        filename, self.args.dir_edge
                    )
                )

            edge = imageio.imread(edge_path)
            if edge.ndim == 3 and edge.shape[2] == 3:
                edge = cv2.cvtColor(edge, cv2.COLOR_RGB2GRAY)
            if edge.ndim == 2:
                edge = np.expand_dims(edge, axis=2)

            target_h, target_w = lr_upscaled.shape[:2]
            if edge.shape[0] != target_h or edge.shape[1] != target_w:
                edge_resized = cv2.resize(
                    edge, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
                if edge_resized.ndim == 2:
                    edge_resized = np.expand_dims(edge_resized, axis=2)
                edge = edge_resized
        else:
            # Fallback: compute edge from upscaled LR
            edge = common.compute_canny_edge(lr_upscaled)
        
        # Add edge to upscaled LR
        lr_enhanced = common.add_edge_to_image(lr_upscaled, edge)
        
        lr_t, = common.np2Tensor(lr_enhanced, rgb_range=self.args.rgb_range)

        return lr_t, -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

