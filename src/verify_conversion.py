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
Verifica visualmente la conversión YOLO-OBB dibujando las esquinas almacenadas
en los .txt sobre los frames originales.

Dos modos:
  verify()         → outputs/check/         frames con clases raras + aleatorios
  verify_rotated() → outputs/check_rotated/ frames con cajas a ángulos no-axiales
                     (angle_deg % 90 ∈ [25, 65]), con el ángulo GT anotado

Uso:
    python -m src.verify_conversion \
        --csv train.csv \
        --frames data/raw/train_frames \
        --labels data/yolo/labels \
        --out outputs/check \
        --out-rotated outputs/check_rotated \
        --n 8 --n-rotated 6 --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

IMG_W = 1920
IMG_H = 1080

CLASS_NAMES = [
    "auto", "combi", "microbus", "minibus", "omnibus",
    "articulado", "camion", "mototaxi", "motocicleta",
]

CLASS_COLORS = [
    (0,   210,  0),    # 0 auto        verde
    (255, 140,  0),    # 1 combi        naranja
    (0,   80,  255),   # 2 microbus     azul
    (255, 0,   120),   # 3 minibus      rosa
    (220, 220,  0),    # 4 omnibus      amarillo
    (255, 0,   255),   # 5 articulado   magenta  ← raro
    (200, 80,   0),    # 6 camion       naranja oscuro
    (160, 0,   220),   # 7 mototaxi     violeta  ← raro
    (0,  200,  200),   # 8 motocicleta  cyan
]

RARE_CLASS_IDS = {4, 5, 7}  # 0-indexed: omnibus, articulado, mototaxi

# Ángulo considerado "rotado": distancia a múltiplo de 90° > ROT_THRESHOLD
ROT_THRESHOLD = 25.0


class DetInfo(NamedTuple):
    """Información de una detección del CSV, emparejada por índice con el .txt."""
    cat_id: int      # 1-indexed (CSV original)
    angle_deg: float
    is_rotated: bool


def _dist_to_axis(angle_deg: float) -> float:
    """Distancia mínima del ángulo al múltiplo de 90° más cercano [0..45]."""
    return min(angle_deg % 90, 90 - angle_deg % 90)


def _is_rotated(angle_deg: float, threshold: float = ROT_THRESHOLD) -> bool:
    return _dist_to_axis(angle_deg) >= threshold


def load_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def read_label(txt_path: Path) -> list[tuple[int, list[float]]]:
    """Lee un .txt YOLO-OBB → [(class_id, [x1,y1,x2,y2,x3,y3,x4,y4]), ...]."""
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return []
    detections = []
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        detections.append((int(parts[0]), [float(p) for p in parts[1:]]))
    return detections


def _parse_csv_dets(target: str) -> list[DetInfo]:
    """Parsea una celda Target de train.csv → lista de DetInfo (mismo orden que .txt)."""
    if target.strip().lower() == "none":
        return []
    result = []
    for tok in target.strip().split(";"):
        parts = tok.strip().split()
        if len(parts) != 6:
            continue
        cat_id = int(parts[0])
        w, h   = float(parts[3]), float(parts[4])
        angle  = float(parts[5])
        # Filtra las mismas cajas que convert_to_yolo_obb descarta
        if w < 1.0 or h < 1.0:
            continue
        result.append(DetInfo(cat_id, angle, _is_rotated(angle)))
    return result


def draw_obb(
    draw: ImageDraw.ImageDraw,
    coords_norm: list[float],
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
    line_width: int = 3,
) -> None:
    """Dibuja un OBB (4 esquinas normalizadas [0,1]) y su etiqueta."""
    pts = [
        (coords_norm[i] * IMG_W, coords_norm[i + 1] * IMG_H)
        for i in range(0, 8, 2)
    ]
    for i in range(4):
        draw.line([pts[i], pts[(i + 1) % 4]], fill=color, width=line_width)

    tx, ty = pts[0]
    bbox = font.getbbox(label) if hasattr(font, "getbbox") else (0, 0, len(label) * 8, 14)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([tx, ty - th - 4, tx + tw + 4, ty], fill=(0, 0, 0, 180))
    draw.text((tx + 2, ty - th - 2), label, fill=color, font=font)


# ── Selección y dibujo para verify() ─────────────────────────────────────────

def select_frames(
    csv_path: Path,
    labels_dir: Path,
    n_total: int = 8,
    n_rare: int = 3,
    seed: int = 42,
) -> list[str]:
    random.seed(seed)
    df = pd.read_csv(csv_path)
    id_col  = next(c for c in df.columns if c.lower() == "id")
    tgt_col = next(c for c in df.columns if c.lower() == "target")
    df = df.rename(columns={id_col: "Id", tgt_col: "Target"})

    frames_with_rare: list[str] = []
    frames_with_dets: list[str] = []

    for row in df.itertuples(index=False):
        t = str(row.Target).strip().lower()
        if t == "none":
            continue
        frames_with_dets.append(row.Id)
        classes = {int(p.split()[0]) - 1 for p in t.split(";") if len(p.split()) == 6}
        if classes & RARE_CLASS_IDS:
            frames_with_rare.append(row.Id)

    selected_rare   = random.sample(frames_with_rare, min(n_rare, len(frames_with_rare)))
    remaining_pool  = [f for f in frames_with_dets if f not in set(selected_rare)]
    selected_random = random.sample(remaining_pool, min(n_total - len(selected_rare), len(remaining_pool)))
    selected = selected_rare + selected_random

    print(f"Frames seleccionados: {len(selected_rare)} con clase rara + {len(selected_random)} aleatorios")
    for fid in selected:
        dets = read_label(labels_dir / f"{fid}.txt")
        print(f"  {fid}  clases: {', '.join(sorted({CLASS_NAMES[d[0]] for d in dets}))}")
    return selected


def verify(
    csv_path: Path,
    frames_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    n: int = 8,
    seed: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(18)
    frame_ids = select_frames(csv_path, labels_dir, n_total=n, n_rare=3, seed=seed)

    for frame_id in frame_ids:
        img_path = frames_dir / f"{frame_id}.jpg"
        if not img_path.exists():
            print(f"  [SKIP] {img_path}")
            continue
        dets = read_label(labels_dir / f"{frame_id}.txt")
        img  = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        for class_id, coords in dets:
            draw_obb(draw, coords, CLASS_COLORS[class_id], CLASS_NAMES[class_id], font)
        out_path = out_dir / f"{frame_id}_check.jpg"
        img.save(out_path, quality=92)
        print(f"  Guardado: {out_path.name}  ({len(dets)} dets)")

    print(f"\n{len(frame_ids)} imágenes guardadas en {out_dir}/")


# ── Selección y dibujo para verify_rotated() ─────────────────────────────────

def select_rotated_frames(
    csv_path: Path,
    n: int = 6,
    seed: int = 99,
) -> list[tuple[str, list[DetInfo]]]:
    """Devuelve n frames con ≥1 caja rotada (angle % 90 ≥ ROT_THRESHOLD).

    Garantiza diversidad de videos: como máximo 1 frame por video.
    Ordena candidatos por máxima distancia al eje para priorizar los más rotados.
    """
    random.seed(seed)
    df = pd.read_csv(csv_path)
    id_col  = next(c for c in df.columns if c.lower() == "id")
    tgt_col = next(c for c in df.columns if c.lower() == "target")
    df = df.rename(columns={id_col: "Id", tgt_col: "Target"})

    # Candidatos: (max_dist_to_axis, frame_id, video_id, dets)
    candidates: list[tuple[float, str, str, list[DetInfo]]] = []
    for row in df.itertuples(index=False):
        dets = _parse_csv_dets(str(row.Target))
        rotated_dets = [d for d in dets if d.is_rotated]
        if not rotated_dets:
            continue
        max_dist = max(_dist_to_axis(d.angle_deg) for d in rotated_dets)
        video_id = row.Id.rsplit("_", 1)[0]
        candidates.append((max_dist, row.Id, video_id, dets))

    # Ordenar de más rotado a menos y garantizar un frame por video
    candidates.sort(key=lambda x: -x[0])
    seen_videos: set[str] = set()
    selected: list[tuple[str, list[DetInfo]]] = []
    for max_dist, frame_id, video_id, dets in candidates:
        if video_id in seen_videos:
            continue
        seen_videos.add(video_id)
        selected.append((frame_id, dets))
        if len(selected) == n:
            break

    print(f"\nFrames seleccionados con cajas rotadas (angle%90 >= {ROT_THRESHOLD:.0f} grados):")
    for frame_id, dets in selected:
        rotated = [(d.angle_deg, CLASS_NAMES[d.cat_id - 1]) for d in dets if d.is_rotated]
        rotated_str = ", ".join(f"{cls}@{a:.1f}°" for a, cls in sorted(rotated, key=lambda x: -_dist_to_axis(x[0])))
        print(f"  {frame_id}  [{rotated_str}]")
    return selected


def verify_rotated(
    csv_path: Path,
    frames_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    n: int = 6,
    seed: int = 99,
) -> None:
    """Dibuja frames con cajas rotadas; anota ángulo GT y resalta las cajas rotadas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    font       = load_font(18)
    font_small = load_font(14)

    selected = select_rotated_frames(csv_path, n=n, seed=seed)

    for frame_id, csv_dets in selected:
        img_path = frames_dir / f"{frame_id}.jpg"
        if not img_path.exists():
            print(f"  [SKIP] {img_path}")
            continue

        txt_dets = read_label(labels_dir / f"{frame_id}.txt")
        if len(txt_dets) != len(csv_dets):
            # Mismatch inesperado — dibujar igual pero sin anotación de ángulo
            print(f"  [WARN] {frame_id}: len(txt)={len(txt_dets)} != len(csv)={len(csv_dets)}")
            csv_dets = [DetInfo(d[0] + 1, 0.0, False) for d in txt_dets]

        img  = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")

        for (class_id, coords), det_info in zip(txt_dets, csv_dets):
            color = CLASS_COLORS[class_id]
            if det_info.is_rotated:
                # Caja rotada: línea más gruesa + ángulo GT en la etiqueta
                label      = f"{CLASS_NAMES[class_id]} {det_info.angle_deg:.1f}°"
                line_width = 5
            else:
                label      = CLASS_NAMES[class_id]
                line_width = 2

            draw_obb(draw, coords, color, label, font, line_width=line_width)

        out_path = out_dir / f"{frame_id}_rotated.jpg"
        img.save(out_path, quality=92)
        n_rot = sum(1 for d in csv_dets if d.is_rotated)
        print(f"  Guardado: {out_path.name}  ({len(txt_dets)} dets, {n_rot} rotadas)")

    print(f"\n{len(selected)} imágenes guardadas en {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verifica visualmente la conversión YOLO-OBB"
    )
    parser.add_argument("--csv",         default=Path("train.csv"),              type=Path)
    parser.add_argument("--frames",      default=Path("data/raw/train_frames"),  type=Path)
    parser.add_argument("--labels",      default=Path("data/yolo/labels"),       type=Path)
    parser.add_argument("--out",         default=Path("outputs/check"),          type=Path)
    parser.add_argument("--out-rotated", default=Path("outputs/check_rotated"),  type=Path)
    parser.add_argument("--n",           default=8,   type=int, help="Frames aleatorios/raros")
    parser.add_argument("--n-rotated",   default=6,   type=int, help="Frames con cajas rotadas")
    parser.add_argument("--seed",        default=42,  type=int)
    args = parser.parse_args()

    verify(
        args.csv, args.frames, args.labels, args.out,
        n=args.n, seed=args.seed,
    )
    verify_rotated(
        args.csv, args.frames, args.labels, args.out_rotated,
        n=args.n_rotated, seed=args.seed,
    )
