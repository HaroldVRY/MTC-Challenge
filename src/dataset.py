# SMART Challenge 2026: IA para la Movilidad del Perú
# Copyright (C) 2026 Harold Victor Reyna Yangali
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

"""
Convierte las anotaciones de train.csv al formato YOLO-OBB y organiza el dataset
en la estructura data/yolo/{images,labels}/{train,val}/.

Formato YOLO-OBB por línea en el .txt:
    class_id cx_norm cy_norm w_norm h_norm angle_deg
donde cx,cy,w,h están normalizados por el tamaño de la imagen [0,1]
y angle_deg es el ángulo de rotación en grados.

Uso:
    python -m src.dataset --csv data/raw/train.csv \
                          --frames data/raw/train \
                          --out data/yolo \
                          --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Competition uses 1-indexed category_id; YOLO uses 0-indexed class_id.
CATEGORY_OFFSET = 1
NUM_CLASSES = 9
CLASS_NAMES = [
    "auto", "combi", "microbus", "minibus", "omnibus",
    "articulado", "camion", "mototaxi", "motocicleta",
]


def parse_target(target: str) -> list[list[float]]:
    """Parsea la columna Target de train.csv y devuelve lista de detecciones.

    Cada detección es [category_id, cx, cy, w, h, angle_deg].
    Devuelve lista vacía para "none".
    """
    if target.strip().lower() == "none":
        return []
    detections = []
    for token in target.strip().split(";"):
        parts = token.strip().split()
        if len(parts) != 6:
            log.warning("Token inesperado ignorado: %s", token)
            continue
        cat_id, cx, cy, w, h, angle = parts
        detections.append([
            int(cat_id), float(cx), float(cy),
            float(w), float(h), float(angle),
        ])
    return detections


def convert_to_yolo_obb(
    detections: list[list[float]],
    img_w: int,
    img_h: int,
) -> list[str]:
    """Convierte detecciones en píxeles al formato YOLO-OBB normalizado."""
    lines = []
    for det in detections:
        cat_id, cx, cy, w, h, angle = det
        class_id = int(cat_id) - CATEGORY_OFFSET  # 0-indexed
        lines.append(
            f"{class_id} "
            f"{cx / img_w:.6f} {cy / img_h:.6f} "
            f"{w / img_w:.6f} {h / img_h:.6f} "
            f"{angle:.4f}"
        )
    return lines


def build_dataset(
    csv_path: Path,
    frames_dir: Path,
    out_dir: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Construye el dataset YOLO-OBB a partir del CSV y los frames extraídos."""
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(csv_path)
    log.info("Filas en train.csv: %d", len(df))

    # Detectar tamaño de imagen leyendo un frame de muestra
    # (se asume resolución uniforme; si no, leer por frame)
    import cv2
    sample_frame = next(frames_dir.rglob("*.jpg"), None) or next(frames_dir.rglob("*.png"), None)
    if sample_frame is None:
        raise FileNotFoundError(f"No se encontraron imágenes en {frames_dir}")
    img = cv2.imread(str(sample_frame))
    img_h, img_w = img.shape[:2]
    log.info("Resolución detectada: %dx%d", img_w, img_h)

    # Split train/val por video_id para evitar data leakage temporal
    video_ids = df["Id"].apply(lambda x: x.rsplit("_", 1)[0]).unique().tolist()
    random.shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * val_ratio))
    val_videos = set(video_ids[:n_val])
    log.info("Videos total: %d  |  val: %d  |  train: %d",
             len(video_ids), n_val, len(video_ids) - n_val)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Convirtiendo"):
        frame_id: str = row["Id"]
        target: str = str(row["Target"])
        video_id = frame_id.rsplit("_", 1)[0]
        split = "val" if video_id in val_videos else "train"

        # Buscar el archivo de imagen correspondiente
        img_path = frames_dir / f"{frame_id}.jpg"
        if not img_path.exists():
            img_path = frames_dir / f"{frame_id}.png"
        if not img_path.exists():
            skipped += 1
            continue

        # Copiar imagen
        dst_img = out_dir / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)

        # Escribir etiqueta
        detections = parse_target(target)
        label_lines = convert_to_yolo_obb(detections, img_w, img_h)
        dst_lbl = out_dir / "labels" / split / (img_path.stem + ".txt")
        dst_lbl.write_text("\n".join(label_lines))
        copied += 1

    log.info("Frames procesados: %d  |  omitidos: %d", copied, skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte train.csv a formato YOLO-OBB")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path, help="Directorio con los frames extraídos")
    parser.add_argument("--out", default=Path("data/yolo"), type=Path)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    build_dataset(args.csv, args.frames, args.out, args.val_ratio, args.seed)
