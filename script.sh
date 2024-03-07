
#CUDA_VISIBLE_DEVICES=2
#tensorboard --logdir=/home/sslunder0/project/NextProject/gaussian-splatting --port 6006
#chair || chair_test || chair_bptest || chair_ADC

#python train.py -s /data1/dataset/nerf_synthetic/chair/ --model_path output/chair_adc_only2

python render.py -m output/chair_ADC/ --eval --iteration 30000
python render.py -m output/chair_ADC/ --eval --iteration 7000
python metrics.py -m output/chair_ADC/