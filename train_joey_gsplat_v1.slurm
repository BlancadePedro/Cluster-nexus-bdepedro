#!/bin/bash
#SBATCH --job-name=gsplat_joey_fixed
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/fs/nexus-scratch/bdepedro/outputs/gsplat_joey_fixed/train_log_%j.out
#SBATCH --error=/fs/nexus-scratch/bdepedro/outputs/gsplat_joey_fixed/train_err_%j.err

echo "🚀 Starting training FIXED at $(date) on node: $HOSTNAME"

# CRÍTICO: Cargar GCC 11.2.0 ANTES de CUDA
module load gcc/11.2.0
module load cuda/12.1.1

source ~/.bashrc
conda activate nerf

export PYTHONPATH=/fs/nexus-scratch/bdepedro/gsplat:$PYTHONPATH
export OMP_NUM_THREADS=1
export TORCH_DATALOADER_WORKERS=1

# Mostrar versiones para debug
echo "=== Environment Check ==="
echo "GCC version: $(gcc --version | head -n1)"
echo "CUDA version: $(nvcc --version | grep release)"
echo "Python: $(which python)"
echo "========================"

cd /fs/nexus-scratch/bdepedro/gsplat/examples

# Usar data-factor 4 para mejor calidad
python simple_trainer.py default \
  --data-dir /fs/nexus-scratch/bdepedro/datasets/sitcoms3d/Friends-joey_apartment_processed \
  --result-dir /fs/nexus-scratch/bdepedro/outputs/gsplat_joey_fixed \
  --data-factor 4 \
  --max-steps 30000 \
  --save-ply \
  --ply-steps 15000 30000 \
  --disable-viewer

echo "✅ Training finished at $(date)"