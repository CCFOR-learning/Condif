# Condif
ConDiF: Confidence-guided Direction Fields for Structure-aware Diffusion Inpainting

conda create -n condif python=3.9
conda activate condif

# Install PyTorch 1.12.1 with CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# Install other dependencies
pip install -r requirements.txt
