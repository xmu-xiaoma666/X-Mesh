import sys
from tqdm import tqdm
import torch
import numpy as np
import random
import torchvision
import os
import argparse
from pathlib import Path
import open3d as o3d
from render import Renderer
from utils import device 
from Normalization import MeshNormalizer
import kaolin.ops.mesh
import kaolin as kal
import imageio.v2
imageio.plugins.freeimage.download()
import os.path as osp
from torchvision import transforms
from mesh import Mesh
import os
from PIL import Image
import json
from utils import device 
import itertools
from numpy import *
import copy

        
def capt(li):
    for i in range(len(li)):
        li[i] = li[i][0].upper()+li[i][1:]
    return li

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=0.22)
    parser.add_argument('--data_path', type=str, default='DatasetResult_MIT30')


    
    args = parser.parse_args()

    our_iters = []
    
    obj_cls = ['vase', 'SoldierBoy', 'candle', 'squirrel', 'phoenix', 'lamp', 'castle', 'dragon', 'bird', 'wardrobe', 'cat', 'treefrog', 'robot', 'BunnyHead', 'person', 'BlueWhale', 'horse', 'skull', 'chair', 'alien', 'bed', 'monster', 'Forklift', 'hollow_pig', 'owl', 'sit_tiger', 'Sofa', 'Vanity_Table', 'wooly_sheep', 'chameleon']
    obj_cls = capt(obj_cls)

    for obj in tqdm(obj_cls):
        objdir = os.path.join(args.data_path,obj)
        cls_our_iters = []
        for i in range(5):
            our_dir = os.path.join(objdir,obj+'Our_MIT30/{}'.format(i))
            
            # load prompt，color和normal
            our_cfg_path = os.path.join(our_dir,"similarity_laion400m.json")
            with open(our_cfg_path,'r') as f:
                our_simi = json.load(f)
            

            our_app = False
            for iter, simi in our_simi.items():
                if simi>args.threshold:
                    cls_our_iters.append(eval(iter))
                    our_app = True
                    break
            if(not our_app):
                cls_our_iters.append(2000)
            
        
        cls_our_iter = mean(cls_our_iters)
        print(f"{obj}_our_score",cls_our_iter.item())
        our_iters.append(cls_our_iter)
        
    print(f"final_our_score",mean(our_iters))
