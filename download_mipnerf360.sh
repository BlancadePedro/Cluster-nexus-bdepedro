#!/bin/bash
#
# Script para descargar el dataset Mip-NeRF 360
# Este es el dataset estándar para Gaussian Splatting
#

echo "=========================================="
echo "Descargando Mip-NeRF 360 Dataset"
echo "=========================================="

cd /fs/nexus-scratch/bdepedro

# Activar entorno
source ~/.bashrc
conda activate nerf

# Crear directorio para datasets públicos
mkdir -p /fs/nexus-scratch/bdepedro/datasets/mipnerf360
cd /fs/nexus-scratch/bdepedro/gsplat/examples/datasets

# Descargar el dataset principal (incluye garden, room, kitchen, etc.)
echo ""
echo "Descargando Mip-NeRF 360 dataset principal..."
echo "Esto incluye: garden, bicycle, bonsai, counter, kitchen, room, stump"
echo "Tamaño total: ~20GB"
echo ""

python download_dataset.py \
  --dataset mipnerf360 \
  --save-dir /fs/nexus-scratch/bdepedro/datasets/mipnerf360

echo ""
echo "=========================================="
echo "Descarga completada!"
echo "=========================================="
echo ""
echo "Datasets disponibles en:"
echo "/fs/nexus-scratch/bdepedro/datasets/mipnerf360/360_v2/"
echo ""
echo "Escenas disponibles:"
ls -lh /fs/nexus-scratch/bdepedro/datasets/mipnerf360/360_v2/ 2>/dev/null || echo "Ejecuta este script para descargar"
echo ""
echo "Para entrenar, usa:"
echo "sbatch /fs/nexus-scratch/bdepedro/scripts/gsplat/train_mipnerf_garden.slurm"
