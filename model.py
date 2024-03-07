import torch
import argparse
from nerf.utils import *

opt = argparse.Namespace(H=1080, O=False, W=1920, bg_radius=-1, bound=2, ckpt='latest', clip_text='', color_space='srgb', 
              cuda_ray=False, density_thresh=10, dt_gamma=0.0078125, error_map=False, ff=False, fovy=50, fp16=False, 
              gui=False, iters=30000, lr=0.01, max_ray_batch=4096, max_spp=64, max_steps=1024, min_near=0.2, num_rays=4096, 
              num_steps=512, offset=[0, 0, 0], patch_size=1, path='data/fox', preload=False, radius=5, rand_pose=-1, scale=0.33, 
              seed=0, tcnn=False, test=True, update_extra_interval=16, upsample_steps=0, workspace='trial_nerf')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from nerf.network import NeRFNetwork

print(opt)
    
seed_everything(opt.seed)

model = NeRFNetwork(
    encoding="hashgrid",
    bound=opt.bound,
    cuda_ray=opt.cuda_ray,
    density_scale=1,
    min_near=opt.min_near,
    density_thresh=opt.density_thresh,
    bg_radius=opt.bg_radius,
)

print(model)

criterion = torch.nn.MSELoss(reduction='none')

if opt.test:
    metrics = [PSNRMeter(), LPIPSMeter(device=device)]
    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
    
    print( model.density(torch.tensor([0.1, 0.1, 0.1]).cuda()) )

    x = torch.tensor([[-1.0, -1.0, -1.0]])  
    d = torch.tensor([[0.1, 0.1, 0.1]])     

    geo_feat = model.density(x)['geo_feat']

    color_output = model.color(x, d, geo_feat=geo_feat)

    print('end')
