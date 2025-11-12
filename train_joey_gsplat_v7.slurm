#!/bin/bash
#SBATCH --job-name=gsplat_joey_v7
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=/fs/nexus-scratch/bdepedro/outputs/gsplat_joey_v7/train_log_%j.out
#SBATCH --error=/fs/nexus-scratch/bdepedro/outputs/gsplat_joey_v7/train_err_%j.err

echo "Starting training (GSplat Joey v7) at $(date) on node: $HOSTNAME"

module load gcc/11.2.0
module load cuda/12.1.1

source ~/.bashrc
conda activate nerf

export PYTHONPATH=/fs/nexus-scratch/bdepedro/gsplat:$PYTHONPATH
export OMP_NUM_THREADS=4
export TORCH_DATALOADER_WORKERS=0

echo "=== Environment Check ==="
echo "GCC: $(gcc --version | head -n1)"
echo "CUDA: $(nvcc --version | grep release)"
echo "Python: $(which python)"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "========================"

cd /fs/nexus-scratch/bdepedro/gsplat

python - <<'PYCODE'
import sys, os
sys.path.append("/fs/nexus-scratch/bdepedro/gsplat/examples")
from simple_trainer import Config, main, DefaultStrategy

cfg = Config(
    data_dir="/fs/nexus-scratch/bdepedro/datasets/sitcoms3d/Friends-joey_apartment",
    result_dir="/fs/nexus-scratch/bdepedro/outputs/gsplat_joey_v7",
    data_factor=4,
    max_steps=30000,
    save_ply=True,
    ply_steps=[15000, 30000],
    disable_viewer=True,
    init_scale=0.5,
    init_opa=0.5,
    ssim_lambda=0.2,
    packed=True,
    antialiased=True,
    random_bkgd=True,
    opacity_reg=0.001,
    scale_reg=0.001,
    strategy=DefaultStrategy(verbose=True),
)

main(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
PYCODE

echo "Training finished at $(date)"