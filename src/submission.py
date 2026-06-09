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
Genera el CSV de submission final a partir de las predicciones JSON.

Formato de cada celda Target (si hay detecciones):
    score category_id cx cy width height angle_deg[;score category_id ...]

Si no hay detecciones para un frame: "none" (nunca vacío).

Uso:
    python -m src.submission --preds outputs/tracked_predictions.json \
                             --sample sample_submission.csv \
                             --out outputs/submission.csv \
                             --max-det 300
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def format_detection(det: dict) -> str:
    """Serializa una detección al formato del submission."""
    return (
        f"{det['score']:.4f} {det['category_id']} "
        f"{det['cx']:.2f} {det['cy']:.2f} "
        f"{det['w']:.2f} {det['h']:.2f} "
        f"{det['angle_deg']:.2f}"
    )


def validate_detection(det: dict) -> bool:
    """Verifica que una detección cumpla las restricciones del evaluador."""
    if not (1 <= det["category_id"] <= 9):
        return False
    if det["w"] <= 0 or det["h"] <= 0:
        return False
    if not all(isinstance(det[k], (int, float)) and
               det[k] == det[k]  # NaN check
               for k in ("score", "cx", "cy", "w", "h", "angle_deg")):
        return False
    return True


def build_submission(
    predictions: dict[str, list[dict]],
    sample_df: pd.DataFrame,
    max_det: int = 300,
) -> pd.DataFrame:
    """Construye el DataFrame de submission con los Ids en el mismo orden que el sample."""
    targets = []
    n_none = 0
    for frame_id in sample_df["Id"]:
        dets = predictions.get(frame_id, [])
        valid_dets = [d for d in dets if validate_detection(d)]
        # Cap por score (ya vienen ordenados desc desde inference.py)
        valid_dets = valid_dets[:max_det]
        if not valid_dets:
            targets.append("none")
            n_none += 1
        else:
            targets.append(";".join(format_detection(d) for d in valid_dets))

    log.info("Frames con 'none': %d / %d", n_none, len(sample_df))
    return pd.DataFrame({"Id": sample_df["Id"], "Target": targets})


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description="Genera submission.csv")
    parser.add_argument("--preds", required=True, type=Path,
                        help="JSON con predicciones (de inference.py o tracking.py)")
    parser.add_argument("--sample", default=Path("sample_submission.csv"), type=Path)
    parser.add_argument("--out", default=Path("outputs/submission.csv"), type=Path)
    parser.add_argument("--max-det", default=300, type=int)
    args = parser.parse_args()

    with open(args.preds) as f:
        preds = json.load(f)

    sample_df = pd.read_csv(args.sample)
    submission_df = build_submission(preds, sample_df, args.max_det)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(args.out, index=False, lineterminator="\r\n")
    log.info("Submission guardado en %s  (%d filas)", args.out, len(submission_df))

    # Sanity checks
    assert len(submission_df) == len(sample_df), "Mismatch en número de filas"
    assert list(submission_df["Id"]) == list(sample_df["Id"]), "Mismatch en orden de Ids"
    assert submission_df["Target"].notna().all(), "Hay celdas vacías en Target"
    log.info("Sanity checks OK.")
