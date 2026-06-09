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
Convierte train.csv al formato YOLO-OBB de 4 esquinas.

Formato de salida por línea de .txt:
    class_id x1 y1 x2 y2 x3 y3 x4 y4
donde class_id es 0-indexed (category_id - 1) y todas las coordenadas
están normalizadas a [0,1] (x dividido entre 1920, y entre 1080).

Las esquinas se calculan directamente desde cx,cy,w,h,angle_deg usando la
rotación estándar 2D, lo que hace irrelevante el rango [0,360) del ángulo.
Ultralytics deriva su propio ángulo interno desde los vértices.

Uso:
    python -m src.convert_to_yolo_obb \
        --csv train.csv \
        --out data/yolo/labels
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

IMG_W: float = 1920.0
IMG_H: float = 1080.0
MIN_W: float = 1.0   # píxeles; cajas con w < MIN_W se descartan
MIN_H: float = 1.0   # píxeles; cajas con h < MIN_H se descartan

CLASS_NAMES = {
    1: "auto", 2: "combi", 3: "microbus", 4: "minibus", 5: "omnibus",
    6: "articulado", 7: "camion", 8: "mototaxi", 9: "motocicleta",
}


def compute_corners(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> list[tuple[float, float]]:
    """Calcula las 4 esquinas del OBB en píxeles (coordenadas de imagen).

    Orden: TL → TR → BR → BL (antes de la rotación), luego rotadas.
    La rotación sigue la convención estándar 2D (anti-horario en coords matemáticas,
    horario en coords de imagen donde Y apunta hacia abajo).
    """
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    hw, hh = w / 2.0, h / 2.0
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [
        (cx + dx * cos_a - dy * sin_a,
         cy + dx * sin_a + dy * cos_a)
        for dx, dy in local
    ]


def convert(
    csv_path: Path,
    out_dir: Path,
    img_w: float = IMG_W,
    img_h: float = IMG_H,
    min_w: float = MIN_W,
    min_h: float = MIN_H,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    id_col  = next(c for c in df.columns if c.lower() == "id")
    tgt_col = next(c for c in df.columns if c.lower() == "target")
    df = df.rename(columns={id_col: "Id", tgt_col: "Target"})
    log.info("Cargado %s: %d filas", csv_path.name, len(df))

    skipped_degen: Counter[int] = Counter()   # category_id → n cajas descartadas
    n_clamped   = 0   # cajas con ≥1 esquina fuera de imagen (antes del clamp)
    n_written   = 0   # detecciones escritas al final
    n_empty_txt = 0   # .txt vacíos (frames "none" o sin dets válidas)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Convirtiendo"):
        frame_id: str = row.Id
        target: str   = str(row.Target).strip()

        out_path = out_dir / f"{frame_id}.txt"

        if target.lower() == "none":
            out_path.write_text("")
            n_empty_txt += 1
            continue

        lines: list[str] = []
        for tok in target.split(";"):
            parts = tok.strip().split()
            if len(parts) != 6:
                continue  # token malformado, ignorar

            cat_id  = int(parts[0])
            cx      = float(parts[1])
            cy      = float(parts[2])
            w       = float(parts[3])
            h       = float(parts[4])
            angle   = float(parts[5])

            # ── Filtrar cajas degeneradas ──────────────────────────────────
            if w < min_w or h < min_h:
                skipped_degen[cat_id] += 1
                continue

            # ── Calcular esquinas en píxeles ──────────────────────────────
            corners_px = compute_corners(cx, cy, w, h, angle)

            # ── Detectar si alguna esquina está fuera de la imagen ─────────
            out_of_bounds = any(
                x < 0 or x > img_w or y < 0 or y > img_h
                for x, y in corners_px
            )
            if out_of_bounds:
                n_clamped += 1

            # ── Normalizar y clampear a [0, 1] ────────────────────────────
            flat: list[str] = []
            for x, y in corners_px:
                nx = max(0.0, min(1.0, x / img_w))
                ny = max(0.0, min(1.0, y / img_h))
                flat.append(f"{nx:.6f}")
                flat.append(f"{ny:.6f}")

            class_id = cat_id - 1  # 0-indexed para YOLO
            lines.append(f"{class_id} " + " ".join(flat))
            n_written += 1

        out_path.write_text("\n".join(lines))
        if not lines:
            n_empty_txt += 1  # frame con dets, pero todas degeneradas

    # ── Reporte final ──────────────────────────────────────────────────────
    total_skipped = sum(skipped_degen.values())
    print("\n" + "=" * 60)
    print("REPORTE DE CONVERSIÓN")
    print("=" * 60)
    print(f"  .txt escritos:                 {len(df):,}")
    print(f"  .txt vacíos (none + sin dets): {n_empty_txt:,}")
    print(f"  Detecciones escritas:          {n_written:,}")
    print(f"  Cajas descartadas (w<1|h<1):   {total_skipped:,}")
    print(f"  Cajas con esquinas fuera img:  {n_clamped:,}  (clampeadas a [0,1])")
    print()
    if skipped_degen:
        print("  Descartadas por clase (degeneradas):")
        for cat_id in sorted(skipped_degen):
            print(f"    cat {cat_id:2d} {CLASS_NAMES.get(cat_id,'?'):<12}: {skipped_degen[cat_id]:,}")
    else:
        print("  Ninguna caja descartada.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convierte train.csv a formato YOLO-OBB de 4 esquinas"
    )
    parser.add_argument(
        "--csv", default=Path("train.csv"), type=Path,
        help="Ruta a train.csv (default: train.csv en raíz del proyecto)",
    )
    parser.add_argument(
        "--out", default=Path("data/yolo/labels"), type=Path,
        help="Directorio de salida para los .txt (default: data/yolo/labels)",
    )
    args = parser.parse_args()
    convert(args.csv, args.out)
