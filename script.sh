#!bin/bash

#Training CUDA_VISIBLE_DEVICES=2
#python train.py -s /data1/dataset/nerf_synthetic/chair/ --model_path output/chair_2sig

#Rendering
python render.py -m output/chair_2sig/

#Evaluation
#python metrics.py -m output/chair_2sig/