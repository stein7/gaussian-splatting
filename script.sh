
#CUDA_VISIBLE_DEVICES=2

#python train.py -s /data1/dataset/nerf_synthetic/chair/ --model_path output/chair
python render.py -m output/chair/ --eval --iteration 30000
python render.py -m output/chair/ --eval --iteration 7000
python metrics.py -m output/chair/