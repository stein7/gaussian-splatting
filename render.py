#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
#from tqdm import tqdm
import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, nerf_model):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm.tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, nerf_model)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, nerf_model):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        
        # scene.save(30010) #flip # gaussians._xyz[:, 1:] *= -1
        # scene.save(30020) #exc # gaussians._xyz[:, [1, 2]] = gaussians._xyz[:, [2, 1]]
        # scene.save(30030) #matmul # gaussians._xyz = gaussians._xyz @ Rot
        # Rot = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float32, device='cuda')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, nerf_model)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, nerf_model)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    import argparse
    from nerf.utils import *
    from nerf.network import NeRFNetwork
    opt = argparse.Namespace(H=1080, O=False, W=1920, bg_radius=-1, bound=2, ckpt='latest', clip_text='', color_space='srgb', 
                cuda_ray=False, density_thresh=10, dt_gamma=0.0078125, error_map=False, ff=False, fovy=50, fp16=False, 
                gui=False, iters=30000, lr=0.01, max_ray_batch=4096, max_spp=64, max_steps=1024, min_near=0.2, num_rays=4096, 
                num_steps=512, offset=[0, 0, 0], patch_size=1, path='data/fox', preload=False, radius=5, rand_pose=-1, scale=0.33, 
                seed=0, tcnn=False, test=True, update_extra_interval=16, upsample_steps=0, workspace='/home/sslunder0/project/NextProject/gaussian-splatting/nerf/trained_model/trial_chair_scale1')

    nerf_model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    print(nerf_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss(reduction='none')
    metrics = [PSNRMeter(), LPIPSMeter(device=device)]
    trainer = Trainer('ngp', opt, nerf_model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
    
    print( nerf_model.density(torch.tensor([0.1, 0.1, 0.1]).cuda()) )
    x = torch.tensor([[-1.0, -1.0, -1.0]]).cuda()
    d = torch.tensor([[0.1, 0.1, 0.1]]).cuda()     

    geo_feat = nerf_model.density(x)['geo_feat']

    color_output = nerf_model.color(x, d, geo_feat=geo_feat)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, nerf_model)