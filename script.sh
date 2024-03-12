
#CUDA_VISIBLE_DEVICES=2 #Source : /home/sslunder0/project/NextProject/dataset_nerf_synthetic/chair ## /data1/dataset/nerf_synthetic/chair/

CUDA_VISIBLE_DEVICES=1 python3 train.py -s /home/sslunder0/project/NextProject/dataset_nerf_synthetic/chair --model_path output/chair_blender
# python3 render.py -m output/chair/ --eval --iteration 30000
# python3 render.py -m output/chair/ --eval --iteration 7000
# python3 metrics.py -m output/chair/