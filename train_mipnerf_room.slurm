#!/bin/bash
#SBATCH --job-name=gsplat_room
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=/fs/nexus-scratch/bdepedro/outputs/gsplat_room/train_log_%j.out
#SBATCH --error=/fs/nexus-scratch/bdepedro/outputs/gsplat_room/train_err_%j.err

echo "=========================================="
echo "Training Gaussian Splatting on Mip-NeRF 360 Room (Indoor Scene)"
echo "=========================================="
echo "Starting at $(date) on node: $HOSTNAME"

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

mkdir -p /fs/nexus-scratch/bdepedro/outputs/gsplat_room

cd /fs/nexus-scratch/bdepedro/gsplat

# Configuración optimizada para escena de interior
python - <<'PYCODE'
import sys, os
sys.path.append("/fs/nexus-scratch/bdepedro/gsplat/examples")
from simple_trainer import Config, main, DefaultStrategy

cfg = Config(
    # Dataset - Room es una escena de interior similar al apartamento de Joey
    data_dir="/fs/nexus-scratch/bdepedro/datasets/mipnerf360/360_v2/room",
    result_dir="/fs/nexus-scratch/bdepedro/outputs/gsplat_room",
    data_factor=4,

    # Training
    max_steps=30000,
    eval_steps=[7000, 30000],
    save_steps=[7000, 30000],
    save_ply=True,
    ply_steps=[15000, 30000],
    disable_viewer=True,

    # Optimización para interiores
    init_scale=1.0,
    init_opa=0.1,
    ssim_lambda=0.2,

    opacity_reg=0.0,
    scale_reg=0.0,

    packed=True,
    antialiased=True,
    random_bkgd=True,

    strategy=DefaultStrategy(
        verbose=True,
        refine_start_iter=500,
        refine_stop_iter=15000,
        reset_every=3000,
        refine_every=100,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        grow_scale2d=0.05,
        prune_opa=0.005,
        prune_scale3d=0.1,
        prune_scale2d=0.15,
    ),
)

print("\n" + "="*50)
print("CONFIGURACIÓN DE ENTRENAMIENTO - ROOM (INTERIOR)")
print("="*50)
print(f"Dataset: {cfg.data_dir}")
print(f"Output: {cfg.result_dir}")
print(f"Max steps: {cfg.max_steps}")
print("="*50 + "\n")

main(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
PYCODE

echo ""
echo "=========================================="
echo "Training finished at $(date)"
echo "=========================================="
echo "Resultados en: /fs/nexus-scratch/bdepedro/outputs/gsplat_room/"
