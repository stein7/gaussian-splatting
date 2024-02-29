
#CUDA_VISIBLE_DEVICES=2

#chair || chair_test || chair_bptest

#python train.py -s /data1/dataset/nerf_synthetic/chair/ --model_path output/chair
python render.py -m output/chair_bptest/ --eval --iteration 30000
python render.py -m output/chair_bptest/ --eval --iteration 7000
python metrics.py -m output/chair_bptest/