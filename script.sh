
#CUDA_VISIBLE_DEVICES=2 #Source : /home/sslunder0/project/NextProject/dataset_nerf_synthetic/chair ## /data1/dataset/nerf_synthetic/chair/
# CUDA_VISIBLE_DEVICES=1 python3 train.py -s /home/sslunder0/project/NextProject/dataset_nerf_synthetic/chair --model_path output/chair_nerf_hash_2sig \
#     --convert_SHs_python --compute_cov3D_python --encoder hashgrid
# CUDA_VISIBLE_DEVICES=1 python3 train.py -s /home/sslunder0/project/NextProject/dataset_nerf_synthetic/chair --model_path output/chair_nerf_freq_2sig \
#     --convert_SHs_python --compute_cov3D_python --encoder frequency

# CUDA_VISIBLE_DEVICES=1 python3 render.py -m output/chair_nerf_hash_2sig/ --eval --iteration 30000 --convert_SHs_python --compute_cov3D_python
# CUDA_VISIBLE_DEVICES=1 python3 render.py -m output/chair_nerf_hash_2sig/ --eval --iteration 7000 --convert_SHs_python --compute_cov3D_python
# CUDA_VISIBLE_DEVICES=1 python3 metrics.py -m output/chair_nerf_hash_2sig/


CUDA_VISIBLE_DEVICES=1 python3 render.py -m output/chair_nerf_freq_2sig/ --eval --iteration 30000 --convert_SHs_python --compute_cov3D_python
CUDA_VISIBLE_DEVICES=1 python3 render.py -m output/chair_nerf_freq_2sig/ --eval --iteration 7000 --convert_SHs_python --compute_cov3D_python
CUDA_VISIBLE_DEVICES=1 python3 metrics.py -m output/chair_nerf_freq_2sig/