# Entrenamiento con Mip-NeRF 360 Dataset

Este directorio contiene scripts optimizados para entrenar Gaussian Splatting en el dataset **Mip-NeRF 360**, el estándar de oro para reconstrucción 3D.

## Paso 1: Descargar el Dataset

```bash
cd /fs/nexus-scratch/bdepedro/scripts/gsplat
bash download_mipnerf360.sh
```

**Tamaño:** ~20GB
**Tiempo de descarga:** 15-30 minutos (dependiendo de la conexión)

El dataset incluye estas escenas:
- **garden** - Jardín (exterior) ⭐ Recomendado para empezar
- **room** - Habitación de oficina (interior) ⭐ Recomendado
- **kitchen** - Cocina (interior)
- **counter** - Mostrador (interior)
- **bonsai** - Planta bonsái (objeto)
- **bicycle** - Bicicleta (exterior)
- **stump** - Tocón (exterior)
- **flowers** - Flores (exterior)
- **treehill** - Colina (exterior)

## Paso 2: Entrenar en una Escena

### Opción A: Garden (Escena Exterior - Más Popular)

```bash
sbatch train_mipnerf_garden.slurm
```

**Resultados esperados:**
- PSNR: ~26-27 dB (excelente)
- SSIM: ~0.85-0.87 (muy bueno)
- LPIPS: ~0.15-0.20 (bueno)
- Tiempo: ~3-4 horas
- Gaussians finales: ~1-2M

### Opción B: Room (Escena Interior - Similar a Apartamento)

```bash
sbatch train_mipnerf_room.slurm
```

**Resultados esperados:**
- PSNR: ~30-31 dB (excelente)
- SSIM: ~0.90-0.92 (excelente)
- LPIPS: ~0.12-0.18 (muy bueno)
- Tiempo: ~3-4 horas
- Gaussians finales: ~1-2M

## Paso 3: Ver Resultados

### Métricas de Evaluación

```bash
# Garden
cat /fs/nexus-scratch/bdepedro/outputs/gsplat_garden/stats/val_step30000.json

# Room
cat /fs/nexus-scratch/bdepedro/outputs/gsplat_room/stats/val_step30000.json
```

### Imágenes Renderizadas

```bash
# Garden
ls /fs/nexus-scratch/bdepedro/outputs/gsplat_garden/renders/

# Room
ls /fs/nexus-scratch/bdepedro/outputs/gsplat_room/renders/
```

### Videos de Trayectoria

```bash
# Garden
ls /fs/nexus-scratch/bdepedro/outputs/gsplat_garden/videos/

# Room
ls /fs/nexus-scratch/bdepedro/outputs/gsplat_room/videos/
```

## Comparación con Joey Apartment

| Métrica | Joey (v7) | Garden (esperado) | Room (esperado) |
|---------|-----------|-------------------|-----------------|
| PSNR    | 14.43 dB ❌ | ~27 dB ✅ | ~31 dB ✅ |
| SSIM    | 0.361 ❌ | ~0.86 ✅ | ~0.91 ✅ |
| LPIPS   | 0.551 ❌ | ~0.17 ✅ | ~0.15 ✅ |
| Gaussians | 624K | ~1.5M | ~1.2M |
| Calidad Visual | Ghosting severo | Excelente | Excelente |

## Configuración de los Scripts

Ambos scripts usan configuración optimizada:

```python
- data_factor=4          # Resolución 1/4 (cambiar a 2 o 1 para mayor calidad)
- max_steps=30000        # 30K iteraciones estándar
- packed=True            # Renderizado eficiente
- antialiased=True       # Reduce aliasing
- random_bkgd=True       # Mejora optimización
- init_scale=1.0         # Escala inicial estándar
- init_opa=0.1           # Opacidad baja
- ssim_lambda=0.2        # Balance L1+SSIM
```

## Modificar para Mayor Calidad

Para obtener resultados de paper-quality, edita los scripts y cambia:

```python
data_factor=2,           # o 1 para resolución completa
max_steps=50000,         # Más entrenamiento
```

**Nota:** Esto aumentará significativamente el tiempo de entrenamiento (6-10 horas) y uso de memoria.

## Siguiente Paso

Una vez que veas resultados excelentes con Mip-NeRF 360, podrás aplicar las mismas configuraciones a otros datasets propios capturados correctamente con COLMAP.

## Troubleshooting

### Error: "Dataset not found"
```bash
# Verifica que descargaste el dataset:
ls /fs/nexus-scratch/bdepedro/datasets/mipnerf360/360_v2/
```

### Memoria insuficiente
```bash
# Reduce la resolución en el script:
data_factor=8,  # En lugar de 4
```

### Quiero entrenar en otra escena

Edita el script y cambia la línea:
```python
data_dir="/fs/nexus-scratch/bdepedro/datasets/mipnerf360/360_v2/NOMBRE_ESCENA"
```

Opciones: garden, room, kitchen, counter, bonsai, bicycle, stump, flowers, treehill
