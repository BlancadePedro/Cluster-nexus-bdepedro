# Ejemplos de Resultados

Este directorio contiene ejemplos representativos de los resultados obtenidos en los entrenamientos de Gaussian Splatting.

## Estructura

```
examples_results/
├── garden/              # Mip-NeRF 360 Garden (dataset profesional)
│   ├── metrics_step7000.json
│   ├── render_001.png
│   └── render_002.png
└── joey_v7/            # Joey Apartment v7 (para comparación)
    ├── metrics_final.json
    ├── render_001.png
    └── render_002.png
```

## Comparación de Resultados

### Mip-NeRF 360 Garden (Step 7000/30000)

**Métricas:**
- PSNR: 26.28 dB ✅
- SSIM: 0.825 ✅
- LPIPS: 0.137 ✅
- Tiempo de render: 0.014s/imagen
- Gaussians: ~3.3M

**Estado:** Entrenamiento en progreso (23% completado)
**Calidad visual:** Excelente, sin ghosting

---

### Joey Apartment v7 (Final - Step 30000)

**Métricas:**
- PSNR: 14.43 dB ❌
- SSIM: 0.361 ❌
- LPIPS: 0.551 ❌
- Gaussians: 624K

**Problema:** Dataset inadecuado (frames de TV con calibración inconsistente)
**Calidad visual:** Ghosting severo, artefactos visibles

---

## Interpretación de Métricas

### PSNR (Peak Signal-to-Noise Ratio)
- **Más alto = mejor**
- < 20 dB: Mala calidad
- 20-30 dB: Calidad aceptable
- 30-40 dB: Excelente calidad
- > 40 dB: Calidad excepcional

### SSIM (Structural Similarity Index)
- **Más alto = mejor** (rango 0-1)
- < 0.5: Mala similitud estructural
- 0.5-0.8: Similitud moderada
- 0.8-0.95: Alta similitud
- > 0.95: Excelente similitud

### LPIPS (Learned Perceptual Image Patch Similarity)
- **Más bajo = mejor**
- < 0.1: Diferencias imperceptibles
- 0.1-0.3: Diferencias perceptibles pero aceptables
- > 0.3: Diferencias significativas

---

## Nota sobre Datasets Completos

Los datasets completos y outputs NO están incluidos en este repositorio (28.7GB total):
- **Datasets:** Descargar con `bash download_mipnerf360.sh`
- **Outputs:** Se generan al ejecutar los entrenamientos

Este directorio solo contiene ejemplos representativos para demostración y comparación.
