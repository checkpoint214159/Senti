# main script to call training for now

import glob
import hashlib
import time

import cv2
import hydra
import minedojo
import numpy as np
import torch
from hydra import compose, initialize
from mineclip import MineCLIP
from omegaconf import OmegaConf

# import matplotlib.pyplot as plt
from PIL import Image

num_iters = 500
load_resolution = [1072, 1920]
save_video_path = 'outputs/ahhhh.mp4'
task_id = 'harvest_wool_with_shears_and_sheep'

def load_model(**kwargs):
    pass

def simulation(num_iters, resolution, save_video_path, task_id):
    frameSize = (resolution[0], resolution[1])
    env = minedojo.make(task_id=task_id, image_size=frameSize)
    out = cv2.VideoWriter(save_video_path,cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frameSize[1], frameSize[0]))
    prompts = [env.task_prompt]
    images = []

    obs = env.reset()
    for i in range(num_iters):
        act = env.action_space.no_op()
        act[0] = 1    # forward/backward
        if i % 10 == 0:
            act[2] = 1    # jump
        obs, reward, done, info = env.step(act)
        # print('next frame')
        # env.render()
        
        pic = obs['rgb']

        pic = pic.transpose((1, 2, 0))
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        
        img = pic.transpose((2, 0, 1))
        img = img[None, :, :, :]
        img = torch.Tensor(img).to('cuda')
        images.append(img)

        out.write(pic)

    env.close()
    out.release()


if __name__=='__main__':
    simulation(num_iters=num_iters, resolution=load_resolution, save_video_path=save_video_path, task_id=task_id)