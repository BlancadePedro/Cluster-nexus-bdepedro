#!/bin/bash
#SBATCH --job-name=gsplat_garden
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=/fs/nexus-scratch/bdepedro/outputs/gsplat_garden/train_log_%j.out
#SBATCH --error=/fs/nexus-scratch/bdepedro/outputs/gsplat_garden/train_err_%j.err

echo "=========================================="
echo "Training Gaussian Splatting on Mip-NeRF 360 Garden"
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

# Crear directorio de salida
mkdir -p /fs/nexus-scratch/bdepedro/outputs/gsplat_garden

cd /fs/nexus-scratch/bdepedro/gsplat

# Configuración optimizada para Mip-NeRF 360
python - <<'PYCODE'
import sys, os
sys.path.append("/fs/nexus-scratch/bdepedro/gsplat/examples")
from simple_trainer import Config, main, DefaultStrategy

cfg = Config(
    # Dataset
    data_dir="/fs/nexus-scratch/bdepedro/datasets/mipnerf360/360_v2/garden",
    result_dir="/fs/nexus-scratch/bdepedro/outputs/gsplat_garden",
    data_factor=4,  # Resolución 1/4 para entrenamiento rápido (usa 2 o 1 para mejor calidad)

    # Training
    max_steps=30000,
    eval_steps=[7000, 30000],
    save_steps=[7000, 30000],
    save_ply=True,
    ply_steps=[15000, 30000],
    disable_viewer=True,

    # Optimización - Configuración probada para Mip-NeRF 360
    init_scale=1.0,        # Escala inicial estándar
    init_opa=0.1,          # Opacidad baja inicial
    ssim_lambda=0.2,       # Balance L1+SSIM

    # Regularización moderada
    opacity_reg=0.0,       # Sin regularización (dataset limpio)
    scale_reg=0.0,

    # Features avanzadas
    packed=True,           # Renderizado eficiente
    antialiased=True,      # Reduce aliasing
    random_bkgd=True,      # Mejora optimización de opacidad

    # Strategy optimizada
    strategy=DefaultStrategy(
        verbose=True,
        refine_start_iter=500,
        refine_stop_iter=15000,
        reset_every=3000,
        refine_every=100,
        # Parámetros de crecimiento/poda
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        grow_scale2d=0.05,
        prune_opa=0.005,
        prune_scale3d=0.1,
        prune_scale2d=0.15,
    ),
)

print("\n" + "="*50)
print("CONFIGURACIÓN DE ENTRENAMIENTO")
print("="*50)
print(f"Dataset: {cfg.data_dir}")
print(f"Output: {cfg.result_dir}")
print(f"Max steps: {cfg.max_steps}")
print(f"Data factor: {cfg.data_factor}")
print(f"Packed: {cfg.packed}")
print(f"Antialiased: {cfg.antialiased}")
print("="*50 + "\n")

main(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
PYCODE

echo ""
echo "=========================================="
echo "Training finished at $(date)"
echo "=========================================="
echo ""
echo "Resultados disponibles en:"
echo "/fs/nexus-scratch/bdepedro/outputs/gsplat_garden/"
echo ""
echo "Para ver las métricas:"
echo "cat /fs/nexus-scratch/bdepedro/outputs/gsplat_garden/stats/val_step30000.json"
echo ""
echo "Para ver renders:"
echo "ls /fs/nexus-scratch/bdepedro/outputs/gsplat_garden/renders/"
