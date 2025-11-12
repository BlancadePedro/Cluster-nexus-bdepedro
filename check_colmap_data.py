#!/usr/bin/env python3
"""
Script para verificar la calidad de los datos de COLMAP
"""
import sys
sys.path.append("/fs/nexus-scratch/bdepedro/gsplat")

import numpy as np
from examples.datasets.colmap import Parser
import matplotlib.pyplot as plt

# Cargar el parser
parser = Parser(
    data_dir="/fs/nexus-scratch/bdepedro/datasets/sitcoms3d/Friends-joey_apartment",
    factor=4,
    normalize=True,
    test_every=8,
)

print("="*60)
print("📊 DIAGNÓSTICO DE DATOS COLMAP")
print("="*60)

# 1. Número de puntos 3D
num_points = len(parser.points)
print(f"\n✓ Puntos 3D iniciales: {num_points:,}")
if num_points < 10000:
    print("  ⚠️  PROBLEMA: Muy pocos puntos! Deberías tener >50k")
elif num_points < 50000:
    print("  ⚠️  WARNING: Pocos puntos, considera recapturar con más overlap")
else:
    print("  ✓ Cantidad adecuada de puntos")

# 2. Distribución de colores
colors = parser.points_rgb
print(f"\n✓ Rango de colores RGB:")
print(f"  - Min: {colors.min(axis=0)}")
print(f"  - Max: {colors.max(axis=0)}")
mean_colors = colors.mean(axis=0)
print(f"  - Mean: [{mean_colors[0]:.1f}, {mean_colors[1]:.1f}, {mean_colors[2]:.1f}]")

# 3. Distribución espacial de puntos
points = parser.points
bounds = np.array([points.min(axis=0), points.max(axis=0)])
extent = bounds[1] - bounds[0]
print(f"\n✓ Distribución espacial:")
print(f"  - Bounds: {bounds}")
print(f"  - Extent (XYZ): {extent}")
print(f"  - Scene scale: {parser.scene_scale:.4f}")

# 4. Información de cámaras
# Crear split de train/test si el parser no lo hace
all_imgs = parser.image_paths
if hasattr(parser, "test_every"):
    num_test = len(all_imgs) // parser.test_every
else:
    num_test = len(all_imgs) // 8  # o el valor que pusiste al crear el parser
num_train = len(all_imgs) - num_test

print(f"\n✓ Imágenes:")
print(f"  - Train: {num_train}")
print(f"  - Test: {num_test}")
print(f"  - Total: {len(all_imgs)}")

if num_train < 20:
    print("  ⚠️  PROBLEMA: Pocas imágenes de entrenamiento")

# 5. Verificar calibración de cámaras
K_sample = list(parser.Ks_dict.values())[0]
print(f"\n✓ Matriz intrínseca (K) ejemplo:")
print(K_sample)

# 6. Verificar extrínsecas
c2w_sample = parser.camtoworlds[0]
print(f"\n✓ Pose de cámara ejemplo (cam2world):")
print(c2w_sample)

# 7. Calcular distancia promedio entre puntos (importante para init_scale)
from scipy.spatial import cKDTree
tree = cKDTree(points)
distances, _ = tree.query(points, k=4)  # 4 vecinos más cercanos
avg_dist = distances[:, 1:].mean()  # excluir el punto mismo
print(f"\n✓ Distancia promedio entre puntos: {avg_dist:.6f}")
print(f"  - init_scale recomendado: {avg_dist * 1.0:.6f}")

print("\n" + "="*60)
print("🎯 RECOMENDACIONES:")
print("="*60)

# Recomendaciones basadas en el análisis
if num_points < 50000:
    print("\n1. ❌ CRÍTICO: Ejecuta COLMAP de nuevo con:")
    print("   - Más imágenes con mejor overlap")
    print("   - Asegúrate de que las máscaras no eliminen demasiado")
    
if extent.max() > 10:
    print("\n2. ⚠️  Escena muy grande, considera:")
    print("   - Aumentar global_scale")
    print("   - Verificar que normalize=True")

print("\n3. 💡 Parámetros de entrenamiento sugeridos:")
print(f"   --init-scale {max(0.5, avg_dist * 0.5):.4f}")
print(f"   --init-opa 0.5")
print(f"   --ssim-lambda 0.2")

print("\n" + "="*60)