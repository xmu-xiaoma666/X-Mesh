#!/bin/bash

obj_cls=('vase' 'SoldierBoy' 'candle' 'squirrel' 'phoenix' 'lamp' 'castle' 'dragon' 'bird' 'wardrobe' 'cat' 'treefrog' 'robot' 'BunnyHead' 'person' 'BlueWhale' 'horse' 'skull' 'chair' 'alien' 'chameleon' 'bed' 'monster' 'Forklift' 'hollow_pig' 'owl' 'sit_tiger' 'Sofa' 'Vanity_Table' 'wooly_sheep')

for ((j=0;j<${#obj_cls[@]};j++)) do
    echo ${obj_cls[j]};

    file_path="Dataset/prompt_MIT30/${obj_cls[j]}.txt"
    pathdir="output/${obj_cls[j]^}/${obj_cls[j]^}_CLIP4Mesh/"


    declare -a lines=()

    while read line; do
        lines+=("$line")
    done < $file_path

    #run every prompt
    for(( i=0;i<${#lines[@]};i++)) do
        path=$pathdir$i
        echo $path
        CUDA_VISIBLE_DEVICES=0 python main.py --run branch --obj_path "Dataset/mesh_MIT30/${obj_cls[j]}.obj" \
        --output_dir $path \
        --prompt ${lines[i]} --sigma 12.0  --clamp tanh --n_normaugs 4 --n_augs 1 --normmincrop 0.1 --normmaxcrop 0.4 --geoloss --colordepth 2 --normdepth 2 --frontview --frontview_std 4 --clipavg view --lr_decay 0.9 --clamp tanh --normclamp tanh  --maxcrop 1.0 --save_render --seed 23 \
        --n_iter 1200  --learning_rate 0.0005 --normal_learning_rate 0.0005 --standardize --no_pe --symmetry --background 1 1 1

    done;
done;
