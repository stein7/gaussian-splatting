
#CUDA_VISIBLE_DEVICES=2 // /home/sslunder0/project/NextProject/dataset_nerf_synthetic

python3 train.py -s /home/sslunder0/project/NextProject/dataset_nerf_synthetic/chair --model_path output/chair_base_2sig
python3 render.py -m output/chair_base_2sig/ --eval --iteration 30000
python3 render.py -m output/chair_base_2sig/ --eval --iteration 7000
python3 metrics.py -m output/chair_base_2sig/