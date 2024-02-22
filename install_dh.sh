#!/bin/bash

yes | pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
yes | pip install plyfile==0.8.1 tqdm

yes | pip install submodules/diff-gaussian-rasterization
yes | pip install submodules/simple-knn

python3 - << END
import torch

if torch.cuda.is_available():
    print("CUDA is available.")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA is not available.")
END
