# SMART Challenge 2026 — IA para la Movilidad del Perú

Solución al reto de detección vehicular con **Oriented Bounding Boxes (OBB)** del
Ministerio de Transportes y Comunicaciones (MTC) del Perú.
Plataforma: [Kaggle](https://www.kaggle.com/competitions/mtc-smart-challenge-ia-para-la-movilidad-del-peru)

Licencia: **GPL v3** (requerida por las bases del concurso).

---

## El problema

Detectar y clasificar 9 tipos de vehículos en frames de intersecciones urbanas usando
OBB (cajas con rotación). Métrica: **Macro AP-rIoU @ [0.50 : 0.80]** — cada clase
pesa igual, por lo que las clases raras (articulado, mototaxi) son tan importantes como
`auto`.

| ID | Clase       | Frecuencia |
|----|-------------|-----------|
| 1  | auto        | muy alta  |
| 2  | combi       | media     |
| 3  | microbus    | media     |
| 4  | minibus     | baja      |
| 5  | omnibus     | baja      |
| 6  | articulado  | muy baja  |
| 7  | camion      | baja      |
| 8  | mototaxi    | muy baja  |
| 9  | motocicleta | media     |

---

## Arquitectura de la solución

```
train.csv + frames
      │
      ▼
src/dataset.py          ← Convierte anotaciones a formato YOLO-OBB (.txt por frame)
      │                    Split train/val por video_id (sin data leakage temporal)
      ▼
src/train.py            ← Fine-tuning YOLOv8m-OBB / YOLOv11m-OBB desde pesos DOTA
      │                    Class weights + augmentación heavy (mosaic, mixup, copy-paste)
      ▼
src/inference.py        ← Inferencia frame a frame sobre test set (conf=0.15 bajo)
      │
      ▼
src/tracking.py         ← Boost temporal: consistencia intra-clip → sube scores
      │
      ▼
src/submission.py       ← Genera submission.csv con sanity checks de formato
      │
      ▼
outputs/submission.csv
```

---

## Estructura del repositorio

```
MTC-Challenge/
├── configs/
│   ├── dataset.yaml        ← Config del dataset YOLO (path, nc, names)
│   └── train.yaml          ← Hiperparámetros de entrenamiento
├── data/
│   ├── raw/                ← train.csv + frames extraídos (no versionado)
│   └── yolo/               ← Dataset en formato YOLO-OBB (generado por dataset.py)
├── docs/                   ← Dossier técnico y manual de operación
├── notebooks/              ← Experimentos adicionales / EDA
├── outputs/                ← Predicciones y submission.csv (no versionado)
├── runs/                   ← Pesos y logs de YOLO training (no versionado)
├── src/
│   ├── dataset.py          ← Conversión CSV → YOLO-OBB labels
│   ├── train.py            ← Script de entrenamiento
│   ├── inference.py        ← Inferencia sobre test frames
│   ├── tracking.py         ← Post-procesamiento temporal intra-clip
│   ├── evaluate.py         ← Macro AP-rIoU local (validación offline)
│   └── submission.py       ← Generación y validación del CSV final
├── sample_submission.csv   ← Template con los 23 579 frame IDs del test
├── MTC_Challengue.ipynb    ← Notebook principal (EDA + flujo completo)
├── requirements.txt
└── CLAUDE.md               ← Contexto técnico para Claude Code
```

---

## Setup del entorno (Windows, PowerShell)

```powershell
# 1. Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (Kaggle/Colab) instalar con GPU CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-deps  # el resto sin sobreescribir torch
```

> **Nota de GPU**: sin GPU local, el entrenamiento debe correr en Kaggle (T4×2) o
> Google Colab (T4/A100). El notebook `MTC_Challengue.ipynb` está preparado para
> ejecutarse en esos entornos.

---

## Pipeline paso a paso

### 1. Preparar los datos

```powershell
# Extraer frames del zip (ajusta la ruta a tu carpeta de Descargas)
Expand-Archive "$env:USERPROFILE\Downloads\train.zip" -DestinationPath data\raw\train
Expand-Archive "$env:USERPROFILE\Downloads\test.zip"  -DestinationPath data\raw\test

# Convertir anotaciones a formato YOLO-OBB
python -m src.dataset --csv data/raw/train.csv `
                      --frames data/raw/train `
                      --out data/yolo `
                      --val-ratio 0.15
```

### 2. Entrenar (en Kaggle/Colab con GPU)

```bash
python -m src.train --config configs/train.yaml --data configs/dataset.yaml
```

### 3. Inferencia sobre test

```bash
python -m src.inference --weights runs/mtc_obb/weights/best.pt \
                        --test-dir data/raw/test \
                        --ids sample_submission.csv \
                        --out outputs/raw_predictions.json \
                        --conf 0.15 --iou 0.45
```

### 4. Post-procesamiento temporal

```bash
python -m src.tracking --preds outputs/raw_predictions.json \
                       --out outputs/tracked_predictions.json
```

### 5. Generar submission

```bash
python -m src.submission --preds outputs/tracked_predictions.json \
                         --sample sample_submission.csv \
                         --out outputs/submission.csv
```

### 6. Evaluar localmente (validación offline)

```bash
python -m src.evaluate --gt data/raw/train.csv \
                       --preds outputs/val_predictions.json \
                       --ids outputs/val_ids.txt
```

---

## Estrategias clave contra el desbalance de clases

- **Class weights** en `configs/train.yaml`: articulado y mototaxi ponderados ×10.
- **Oversampling explícito** de clips con clases raras (en `src/dataset.py`).
- **Umbral de confianza bajo (0.15)**: maximiza recall; el AP tolera FPs si el
  ranking de scores es correcto.
- **Temporal score boost** (`src/tracking.py`): detecciones consistentes en frames
  vecinos reciben un multiplicador de score, mejorando recall en clases raras
  que aparecen brevemente.

---

## Entregables finales del concurso

- [ ] Código fuente bajo GPL v3 (este repositorio)
- [ ] Memoria Descriptiva / Dossier Técnico (`docs/dossier_tecnico.docx`)
- [ ] Manual de Operación (`docs/manual_operacion.docx`)
- [ ] Reporte CSV de flujos vehiculares (`outputs/vehicle_flow_report.csv`)
