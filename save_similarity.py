# %matplotlib inline
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import make_interp_spline
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
import json



def test(base_prompt,base_obj_path,base_color_path=None,base_normal_path=None):

    # create mesh
    base_mesh = Mesh(base_obj_path)
    MeshNormalizer(base_mesh)()
    base_vertices = copy.deepcopy(base_mesh.vertices)
    MeshNormalizer(base_mesh)()
    
    

    # set default color to gray
    base_default_color = torch.full(size=(base_mesh.vertices.shape[0], 3), fill_value=0.5, device=device)
    base_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(base_default_color.unsqueeze(0),
                                                                base_mesh.faces.to(device))

    
    if (base_color_path is not None and base_normal_path is not None):
        # loadd offset
        base_color = torch.load(base_color_path).to(device)
        base_normals = torch.load(base_normal_path).to(device)

        # set geometry
        base_base_color = torch.full(size=(base_mesh.vertices.shape[0], 3), fill_value=0.5).to(device)
        base_final_color = torch.clamp(base_base_color + base_color, 0, 1).to(device)
        base_mesh.vertices = base_vertices + base_mesh.vertex_normals * base_normals
    else:
        base_base_color = torch.full(size=(base_mesh.vertices.shape[0], 3), fill_value=0.5).to(device)
        base_final_color = torch.clamp(base_base_color, 0, 1).to(device)
        base_mesh.vertices = base_vertices
        
    
    # get textual feature
    base_prompt_token = tokenizer([base_prompt]).to(device)
    base_encoded_text = clip_model.encode_text(base_prompt_token)
    
    
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
            


        # get visual features
        base_stack_img = torch.stack(base_imgs,dim=0)
        base_encoded_img = clip_model.encode_image(base_stack_img)
        
        # calculate similarity
        base_encoded_img /= base_encoded_img.norm(dim=-1, keepdim=True)
        base_encoded_text /= base_encoded_text.norm(dim=-1, keepdim=True)


        base_score = base_encoded_img @ base_encoded_text.T
        base_score = base_score.mean()
        return base_score
    
def capt(li):
    for i in range(len(li)):
        li[i] = li[i][0].upper()+li[i][1:]
    return li



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=tuple, default=('ViT-B-32', 'laion400m_e31'))
    parser.add_argument('--data_path', type=str, default='DatasetResult_MIT30')
    parser.add_argument('--resolution', type=int, default=224)

    
    args = parser.parse_args()

    # load path
    data_path = args.data_path
    model_name = args.model_name
    resolution = args.resolution


    # create renderer
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(np.pi / 4, resolution / resolution).to(device),
        dim=(resolution, resolution))
    
    # loas open clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1], device=device)
    tokenizer = open_clip.get_tokenizer(model_name[0])


    prompts =[]
    base_scores = []
    our_scores = []
    objs = os.listdir(data_path)
    objs.sort()

    obj_cls = ['vase', 'SoldierBoy', 'candle', 'squirrel', 'phoenix', 'lamp', 'castle', 'dragon', 'bird', 'wardrobe', 'cat', 'treefrog', 'robot', 'BunnyHead', 'person', 'BlueWhale', 'horse', 'skull', 'chair', 'alien', 'bed', 'monster', 'Forklift', 'hollow_pig', 'owl', 'sit_tiger', 'Sofa', 'Vanity_Table', 'wooly_sheep', 'chameleon']
    obj_cls = capt(obj_cls)

    for obj in tqdm(obj_cls):
        print(obj)
        objdir = os.path.join(data_path,obj)
        for i in range(5):

            our_dir = os.path.join(objdir,obj+'Our_MIT30/{}'.format(i))
            if(os.path.exists(os.path.join(our_dir,'similarity_laion400m.json'))):
                continue

            cls_our_scores = {}
            for iter in tqdm(range(0,1200,10)):
                
                our_cfg_path = os.path.join(our_dir,"cfg.json")
                with open(our_cfg_path,'r') as f:
                    our_cfg = json.load(f)
                our_prompt = our_cfg['prompt'] 
                our_obj_path = our_cfg['obj']
                our_color_path = os.path.join(our_cfg['output_dir'],'colors',f"colors_{iter}iter.pt")
                our_normal_path = os.path.join(our_cfg['output_dir'],'normals',f"normals_{iter}iter.pt")

                our_score = test(our_prompt,our_obj_path,our_color_path,our_normal_path)
                cls_our_scores[iter] = our_score.item()
                
                
            with open(os.path.join(our_dir,'similarity_lion400m.json'),'w') as f:
                cls_our_scores = json.dumps(cls_our_scores,indent=4)
                f.write(cls_our_scores)
                    
            


