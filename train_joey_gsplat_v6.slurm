#!/bin/bash
#SBATCH --job-name=gsplat_joey_v6
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=/fs/nexus-scratch/bdepedro/outputs/gsplat_joey_v6/train_log_%j.out
#SBATCH --error=/fs/nexus-scratch/bdepedro/outputs/gsplat_joey_v6/train_err_%j.err

echo "Starting training (GSplat Joey v6) at $(date) on node: $HOSTNAME"

module load gcc/11.2.0
module load cuda/12.1.1

source ~/.bashrc
conda activate nerf

# Rutas y variables de entorno
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

PYTHONPATH=$(pwd):$(pwd)/examples python -m examples.simple_trainer \
  --data-dir /fs/nexus-scratch/bdepedro/datasets/sitcoms3d/Friends-joey_apartment \
  --result-dir /fs/nexus-scratch/bdepedro/outputs/gsplat_joey_v6 \
  --data-factor 4 \
  --max-steps 30000 \
  --save-ply \
  --ply-steps 15000 30000 \
  --disable-viewer

echo "Training finished at $(date)"