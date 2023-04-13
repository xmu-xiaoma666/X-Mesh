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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def save_img(base_prompt,base_obj_path,base_color_path,base_normal_path,out_path):

    # Create Mesh
    base_mesh = Mesh(base_obj_path)
    MeshNormalizer(base_mesh)()
    base_vertices = copy.deepcopy(base_mesh.vertices)
    MeshNormalizer(base_mesh)()
    
    

    # set default color to gray
    base_default_color = torch.full(size=(base_mesh.vertices.shape[0], 3), fill_value=0.5, device=device)
    base_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(base_default_color.unsqueeze(0),
                                                                base_mesh.faces.to(device))

    
    # load offset
    base_color = torch.load(base_color_path).to(device)
    base_normals = torch.load(base_normal_path).to(device)

    # set color and geometry
    base_base_color = torch.full(size=(base_mesh.vertices.shape[0], 3), fill_value=0.5).to(device)
    base_final_color = torch.clamp(base_base_color + base_color, 0, 1).to(device)
    base_mesh.vertices = base_vertices + base_mesh.vertex_normals * base_normals

    
    # save info
    prompt = {"prompt":base_prompt}
    with open(os.path.join(out_path,'prompt.json'),'w') as f:
        f.write(json.dumps(prompt))
    
    
    # set angles
    azims = torch.linspace(0, 2 * np.pi, 8+1)[:-1]  # since 0 =360 dont include last element
    elevs = torch.linspace(-np.pi/6, np.pi/6, 3)
    
    with torch.no_grad():
        j=0
        MeshNormalizer(base_mesh)()
        base_imgs = []
        for elev,azim in itertools.product(elevs,azims):
            j=j+1
            # Vertex colorings
            base_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(base_final_color.unsqueeze(0).to(device),
                                                                        base_mesh.faces.to(device))
            img, mask = kal_render.render_single_view(base_mesh, elev=elev,azim=azim, 
                                                    radius=2.5,
                                                    background=torch.tensor([1, 1, 1]).to(device).float(),
                                                    return_mask=True)
            img = img[0]
            mask = mask[0]
            base_imgs.append(img)
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(out_path, f"{j}.png"))
         

def capt(li):
    for i in range(len(li)):
        li[i] = li[i][0].upper()+li[i][1:]
    return li
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--data_path', type=str, default='DatasetResult_MIT30')
    parser.add_argument('--out_path', type=str, default='./Result/AllViews/')


    
    args = parser.parse_args()

    
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(np.pi / 4, args.resolution / args.resolution).to(device),
        dim=(args.resolution, args.resolution))

    base_scores = []
    our_scores = []
    
    obj_cls = ['vase', 'SoldierBoy', 'candle', 'squirrel', 'phoenix', 'lamp', 'castle', 'dragon', 'bird', 'wardrobe', 'cat', 'treefrog', 'robot', 'BunnyHead', 'person', 'BlueWhale', 'horse', 'skull', 'chair', 'alien', 'bed', 'monster', 'Forklift', 'hollow_pig', 'owl', 'sit_tiger', 'Sofa', 'Vanity_Table', 'wooly_sheep', 'chameleon']
    obj_cls = capt(obj_cls)

    for obj in tqdm(obj_cls):
        objdir = os.path.join(args.data_path,obj)
        
        cls_base_scores = []
        cls_our_scores = []
        for i in range(5):
            our_dir = os.path.join(objdir,obj+'Our_MIT30/{}'.format(i))
            out_our_dir = os.path.join(args.out_path,obj+'Our_MIT30/{}'.format(i))

            if(not os.path.exists(out_our_dir)):
                os.makedirs(out_our_dir)

            # load info
            our_cfg_path = os.path.join(our_dir,"cfg.json")
            with open(our_cfg_path,'r') as f:
                our_cfg = json.load(f)
            our_prompt = our_cfg['prompt'] 
            our_obj_path = our_cfg['obj']
            our_color_path = our_cfg['color_path']
            our_normal_path = our_cfg['normal_path']

            
            save_img(our_prompt,our_obj_path,our_color_path,our_normal_path,out_our_dir)
