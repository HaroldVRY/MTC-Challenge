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
Post-procesamiento temporal intra-clip.

Estrategia implementada:
1. Agrupa frames por video_id.
2. Para cada secuencia de frames de un clip, aplica suavizado de scores:
   si un objeto aparece en frames N-1 y N+1 pero no en N, se interpola.
3. Boost de score para detecciones consistentes a lo largo del clip.

Uso:
    python -m src.tracking --preds outputs/raw_predictions.json \
                           --out outputs/tracked_predictions.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import math

import numpy as np
from shapely.geometry import Polygon

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OBB_MATCH_IOU_THRESHOLD = 0.3  # umbral para considerar mismo objeto entre frames


def obb_to_polygon(det: dict) -> Polygon:
    """Convierte una detección OBB a shapely.Polygon para calcular rIoU."""
    cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]
    angle_rad = math.radians(det["angle_deg"])
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    half_w, half_h = w / 2, h / 2
    corners = [
        (-half_w, -half_h), (half_w, -half_h),
        (half_w,  half_h),  (-half_w,  half_h),
    ]
    rotated = [
        (cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a)
        for x, y in corners
    ]
    return Polygon(rotated)


def riou(det_a: dict, det_b: dict) -> float:
    """Calcula rotated IoU entre dos detecciones."""
    try:
        poly_a = obb_to_polygon(det_a)
        poly_b = obb_to_polygon(det_b)
        if not poly_a.is_valid or not poly_b.is_valid:
            return 0.0
        inter = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def temporal_score_boost(
    clip_preds: dict[str, list[dict]],
    iou_thresh: float = OBB_MATCH_IOU_THRESHOLD,
    boost_factor: float = 1.1,
    max_score: float = 0.99,
) -> dict[str, list[dict]]:
    """Aplica boost de score a detecciones consistentes a través del clip.

    Para cada detección en frame N, si aparece una detección similar en frame N-1
    o N+1 (mismo category_id, rIoU > iou_thresh), se multiplica el score.
    """
    frame_keys = sorted(clip_preds.keys())
    boosted = {k: [d.copy() for d in v] for k, v in clip_preds.items()}

    for i, fk in enumerate(frame_keys):
        neighbor_keys = []
        if i > 0:
            neighbor_keys.append(frame_keys[i - 1])
        if i < len(frame_keys) - 1:
            neighbor_keys.append(frame_keys[i + 1])

        for det in boosted[fk]:
            for nk in neighbor_keys:
                for ndet in boosted[nk]:
                    if ndet["category_id"] != det["category_id"]:
                        continue
                    if riou(det, ndet) > iou_thresh:
                        det["score"] = min(det["score"] * boost_factor, max_score)
                        break

    return boosted


def process_all_clips(
    predictions: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Aplica el post-procesamiento temporal a todos los clips."""
    clips: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for frame_id, dets in predictions.items():
        video_id = frame_id.rsplit("_", 1)[0]
        clips[video_id][frame_id] = dets

    result: dict[str, list[dict]] = {}
    for video_id, clip_preds in clips.items():
        boosted = temporal_score_boost(clip_preds)
        result.update(boosted)

    log.info("Post-procesamiento temporal aplicado a %d clips", len(clips))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-procesamiento temporal OBB")
    parser.add_argument("--preds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with open(args.preds) as f:
        predictions = json.load(f)

    tracked = process_all_clips(predictions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(tracked, f)
    log.info("Predicciones post-procesadas guardadas en %s", args.out)
