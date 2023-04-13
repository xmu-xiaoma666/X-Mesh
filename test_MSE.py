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
import open_clip
import json
from utils import device 
import itertools
from numpy import *
import copy
import clip

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def save_gif(dir,fps):
    imgpath = dir
    frames = []
    for idx in sorted(os.listdir(imgpath)):
        img = osp.join(imgpath,idx)
        frames.append(imageio.imread(img))
    imageio.mimsave(os.path.join(dir, 'eval.gif'),frames,'GIF',duration=1/fps)


def save_gif_wo_img(dir,fps,imgs):
    frames = []
    for img in imgs:
        frames.append(img)
    imageio.mimsave(os.path.join(dir, 'eval.gif'),frames,'GIF',duration=1/fps)



def test(out_path):
    prompt_path = os.path.join(out_path,'prompt.json')
    with open(prompt_path,'r') as f:
        content = json.load(f)
        prompt = content['prompt']
    base_prompt_token = tokenizer([prompt]).to(device=device)
    base_encoded_text = clip_model.encode_text(base_prompt_token)

    imgs = []
    for i in range(1,25):
        img_path = os.path.join(out_path,str(i)+'.png')
        image = preprocess(Image.open(img_path))
        imgs.append(image)

    # get visual features
    base_stack_img = torch.stack(imgs,dim=0).to(device=device)
    base_encoded_img = clip_model.encode_image(base_stack_img)

    # get similarity
    base_encoded_img /= base_encoded_img.norm(dim=-1, keepdim=True)
    base_encoded_text /= base_encoded_text.norm(dim=-1, keepdim=True)
    base_score = 100*base_encoded_img @ base_encoded_text.T
    base_score = base_score.mean()
    return base_score

def capt(li):
    for i in range(len(li)):
        li[i] = li[i][0].upper()+li[i][1:]
    return li
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=tuple, default=('ViT-B-32', 'laion400m_e31'))
    parser.add_argument('--out_path', type=str, default='./Result/AllViews/')

    
    args = parser.parse_args()


    # load open clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(args.model_name[0], pretrained=args.model_name[1],device=device)
    tokenizer = open_clip.get_tokenizer(args.model_name[0])

    our_scores = []
    
    obj_cls = ['vase', 'SoldierBoy', 'candle', 'squirrel', 'phoenix', 'lamp', 'castle', 'dragon', 'bird', 'wardrobe', 'cat', 'treefrog', 'robot', 'BunnyHead', 'person', 'BlueWhale', 'horse', 'skull', 'chair', 'alien', 'bed', 'monster', 'Forklift', 'hollow_pig', 'owl', 'sit_tiger', 'Sofa', 'Vanity_Table', 'wooly_sheep', 'chameleon']
    obj_cls = capt(obj_cls)

    for obj in tqdm(obj_cls):
        cls_our_scores = []
        for i in range(5):
            out_our_dir = os.path.join(args.out_path,obj+'Our_MIT30/{}'.format(i))

            with torch.no_grad(), torch.cuda.amp.autocast():
                our_score=test(out_our_dir)
            cls_our_scores.append(our_score.item())
        
        cls_our_score = mean(cls_our_scores)
        print(f"{obj}_our_score",cls_our_score.item())
        our_scores.append(cls_our_score)
        
    print(f"final_our_score",mean(our_scores))