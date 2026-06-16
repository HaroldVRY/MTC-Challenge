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
Réplica local de la métrica oficial: Macro AP-rIoU @ [0.50 : 0.80] (step 0.05).

A diferencia de src/evaluate.py (que consume predicciones en JSON intermedio),
este módulo recibe predicciones EXACTAMENTE en el formato de submission, lo
que permite validar un submission.csv generado por inferencia directamente
contra el ground truth de validación, sin pasos intermedios.

Formato de las celdas "Target":
    Ground truth (sin score) : "category_id cx cy width height angle_deg[;...]"
    Predicciones (con score) : "score category_id cx cy width height angle_deg[;...]"
    Frame sin objetos        : "none"

Reglas de matching (por clase, por separado):
    - Las predicciones se ordenan globalmente por score descendente.
    - Cada GT puede emparejarse a lo más una vez (greedy).
    - Una predicción sin GT libre con rIoU >= umbral es FP (incluye
      duplicados sobre un GT ya emparejado y predicciones de clase
      incorrecta, que nunca compiten contra GT de otra clase).

Uso:
    python -m src.eval_metric --gt data/raw/train.csv \
                               --preds outputs/submission_val.csv \
                               --ids configs/val_ids.txt

    python -m src.eval_metric --selftest
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

IOU_THRESHOLDS = [round(t, 2) for t in np.arange(0.50, 0.85, 0.05)]
NUM_CLASSES = 9
CLASS_IDS = list(range(1, NUM_CLASSES + 1))

Detection = tuple[float, float, float, float, float]  # cx, cy, w, h, angle_deg


def obb_to_polygon(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Polygon:
    """Reconstruye el polígono rotado (4 esquinas) de un OBB."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    hw, hh = w / 2.0, h / 2.0
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return Polygon([(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners])


def rotated_iou(box_a: Detection, box_b: Detection) -> float:
    """Rotated IoU (rIoU) entre dos OBBs [cx, cy, w, h, angle_deg]."""
    try:
        pa, pb = obb_to_polygon(*box_a), obb_to_polygon(*box_b)
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        inter = pa.intersection(pb).area
        if inter <= 0.0:
            return 0.0
        union = pa.union(pb).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def parse_target(target: object, has_score: bool) -> list[dict]:
    """Parsea una celda 'Target' (formato submission o GT) a lista de detecciones.

    has_score=True  -> "score category_id cx cy w h angle_deg" (predicciones)
    has_score=False -> "category_id cx cy w h angle_deg"        (ground truth)
    """
    target = str(target).strip()
    if not target or target.lower() == "none":
        return []
    expected_len = 7 if has_score else 6
    dets = []
    for token in target.split(";"):
        parts = token.strip().split()
        if len(parts) != expected_len:
            continue
        try:
            if has_score:
                score = float(parts[0])
                cat_id = int(parts[1])
                box = tuple(float(x) for x in parts[2:])
            else:
                score = 1.0
                cat_id = int(parts[0])
                box = tuple(float(x) for x in parts[1:])
        except ValueError:
            continue
        dets.append({"score": score, "category_id": cat_id, "box": box})
    return dets


def load_csv(path: Path, has_score: bool) -> dict[str, list[dict]]:
    """Carga un CSV (columnas Id, Target) en formato submission a {frame_id: [detecciones]}."""
    df = pd.read_csv(path)
    assert "Id" in df.columns and "Target" in df.columns, f"{path} debe tener columnas Id,Target"
    return {str(row["Id"]): parse_target(row["Target"], has_score=has_score) for _, row in df.iterrows()}


def match_class_at_threshold(
    gt_by_frame: dict[str, list[Detection]],
    preds_by_frame: dict[str, list[tuple[float, Detection]]],
    iou_thresh: float,
) -> tuple[list[int], list[int], int]:
    """Empareja predicciones <-> GT de UNA clase a un umbral de rIoU dado.

    Retorna (tp_flags, fp_flags, total_gt), ambas listas ordenadas por score
    de predicción descendente (orden global, no por frame).
    """
    total_gt = sum(len(v) for v in gt_by_frame.values())

    all_preds: list[tuple[float, str, int]] = []
    for frame_id, dets in preds_by_frame.items():
        for local_idx, (score, _box) in enumerate(dets):
            all_preds.append((score, frame_id, local_idx))
    all_preds.sort(key=lambda x: -x[0])

    matched_gt: dict[str, set[int]] = defaultdict(set)
    tps, fps = [], []
    for score, frame_id, local_idx in all_preds:
        pred_box = preds_by_frame[frame_id][local_idx][1]
        gts = gt_by_frame.get(frame_id, [])
        best_iou, best_j = -1.0, -1
        for j, gt_box in enumerate(gts):
            if j in matched_gt[frame_id]:
                continue
            iou_val = rotated_iou(pred_box, gt_box)
            if iou_val > best_iou:
                best_iou, best_j = iou_val, j
        if best_j >= 0 and best_iou >= iou_thresh:
            matched_gt[frame_id].add(best_j)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    return tps, fps, total_gt


def compute_ap(tps: list[int], fps: list[int], total_gt: int) -> float:
    """AP por interpolación de 101 puntos (estilo COCO), a partir de TP/FP ordenados por score."""
    if total_gt == 0:
        return float("nan")
    if not tps:
        return 0.0
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recalls >= t
        ap += precisions[mask].max() if mask.any() else 0.0
    return ap / 101.0


def evaluate(
    gt: dict[str, list[dict]],
    preds: dict[str, list[dict]],
    frame_ids: list[str] | None = None,
) -> dict:
    """Calcula Macro AP-rIoU @ [0.50:0.80] sobre frame_ids dados.

    gt / preds: {frame_id: [{"score":..., "category_id":..., "box": (cx,cy,w,h,angle_deg)}, ...]}
    (score de GT es irrelevante, se ignora). El matching se hace siempre
    dentro de la misma clase: una predicción con category_id incorrecto
    nunca se compara contra GT de otra clase, por lo que es FP automático
    salvo que exista (coincidencialmente) un objeto de esa clase en ese lugar.

    Retorna:
        {
          "macro_ap": float,
          "per_class": {cat_id: {"ap": float, "per_threshold": {umbral: ap}}},
        }
    """
    if frame_ids is None:
        frame_ids = sorted(set(gt) | set(preds))

    gt_by_class_frame: dict[int, dict[str, list[Detection]]] = {c: defaultdict(list) for c in CLASS_IDS}
    for fid in frame_ids:
        for det in gt.get(fid, []):
            cat = det["category_id"]
            if cat in gt_by_class_frame:
                gt_by_class_frame[cat][fid].append(det["box"])

    preds_by_class_frame: dict[int, dict[str, list[tuple[float, Detection]]]] = {
        c: defaultdict(list) for c in CLASS_IDS
    }
    for fid in frame_ids:
        for det in preds.get(fid, []):
            cat = det["category_id"]
            if cat in preds_by_class_frame:
                preds_by_class_frame[cat][fid].append((det["score"], det["box"]))

    per_class: dict[int, dict] = {}
    class_aps = []
    for cat in CLASS_IDS:
        per_threshold = {}
        for thr in IOU_THRESHOLDS:
            tps, fps, total_gt = match_class_at_threshold(
                gt_by_class_frame[cat], preds_by_class_frame[cat], thr
            )
            per_threshold[thr] = compute_ap(tps, fps, total_gt)
        valid_aps = [v for v in per_threshold.values() if not math.isnan(v)]
        class_ap = float(np.mean(valid_aps)) if valid_aps else 0.0
        per_class[cat] = {"ap": class_ap, "per_threshold": per_threshold}
        class_aps.append(class_ap)
        log.info("Clase %d: AP=%.4f", cat, class_ap)

    macro_ap = float(np.mean(class_aps))
    log.info("Macro AP-rIoU @ [0.50:0.80] = %.4f", macro_ap)
    return {"macro_ap": macro_ap, "per_class": per_class}


# ──────────────────────────────────────────────────────────────────────────
# Caso sintético de control
# ──────────────────────────────────────────────────────────────────────────

def _run_self_test() -> None:
    """Valida rotated_iou, compute_ap y evaluate() contra valores calculados a mano."""

    # 1) rotated_iou: casos con solución geométrica cerrada conocida.
    box = (0.0, 0.0, 10.0, 10.0, 0.0)
    assert abs(rotated_iou(box, box) - 1.0) < 1e-9, "IoU de un OBB consigo mismo debe ser 1.0"

    far_box = (1000.0, 1000.0, 10.0, 10.0, 0.0)
    assert rotated_iou(box, far_box) == 0.0, "OBBs sin solapamiento deben dar IoU=0"

    # Dos cuadrados 10x10 desplazados 5 en x: overlap=50, union=150 -> IoU=1/3
    shifted = (5.0, 0.0, 10.0, 10.0, 0.0)
    iou_shifted = rotated_iou(box, shifted)
    assert abs(iou_shifted - (1.0 / 3.0)) < 1e-6, f"IoU esperado 1/3, obtenido {iou_shifted}"

    # 2) compute_ap: TP/FP conocidos -> AP calculado a mano.
    #    tps=[1,0,0], fps=[0,1,1], total_gt=2
    #    recalls=[0.5,0.5,0.5]  precisions=[1.0, 0.5, 0.333..]
    #    AP = 51/101 (51 puntos de recall en [0, 0.50] con precision máxima 1.0)
    ap_manual = compute_ap([1, 0, 0], [0, 1, 1], total_gt=2)
    expected_ap = 51.0 / 101.0
    assert abs(ap_manual - expected_ap) < 1e-9, f"AP esperado {expected_ap}, obtenido {ap_manual}"

    # 3) evaluate(): escenario sintético con TP, FN, FP por duplicado, FP por
    #    ubicación incorrecta y una clase con match perfecto.
    #
    #    Clase 1 ("auto"):
    #      frame f1: 1 GT con match casi perfecto (score alto) + 1 duplicado
    #                de menor score sobre el mismo GT -> ese duplicado es FP.
    #      frame f2: 1 GT sin ninguna predicción que lo cubra (FN) + 1 FP
    #                en una ubicación completamente distinta.
    #      -> total_gt=2, tps=[1,0,0], fps=[0,1,1] en TODOS los umbrales
    #         (el match de f1 tiene rIoU~1.0, > 0.80) => AP = 51/101 en cada umbral.
    #
    #    Clase 2 ("combi"):
    #      frame f3: 1 GT con match perfecto -> AP = 1.0 en todos los umbrales.
    #
    #    Clases 3-9: sin GT ni predicciones -> AP = 0.0 (no contribuyen señal).
    gt = {
        "f1": [{"score": 1.0, "category_id": 1, "box": (100.0, 100.0, 50.0, 30.0, 0.0)}],
        "f2": [{"score": 1.0, "category_id": 1, "box": (300.0, 300.0, 40.0, 20.0, 45.0)}],
        "f3": [{"score": 1.0, "category_id": 2, "box": (700.0, 700.0, 60.0, 60.0, 90.0)}],
    }
    preds = {
        "f1": [
            {"score": 0.90, "category_id": 1, "box": (100.0, 100.0, 50.0, 30.0, 0.0)},   # TP
            {"score": 0.10, "category_id": 1, "box": (100.0, 100.0, 50.0, 30.0, 0.0)},   # duplicado -> FP
        ],
        "f2": [
            {"score": 0.80, "category_id": 1, "box": (500.0, 500.0, 10.0, 10.0, 0.0)},   # FP (sin solape)
        ],
        "f3": [
            {"score": 0.99, "category_id": 2, "box": (700.0, 700.0, 60.0, 60.0, 90.0)},  # TP
        ],
    }
    frame_ids = ["f1", "f2", "f3"]

    result = evaluate(gt, preds, frame_ids)

    ap_class1 = result["per_class"][1]["ap"]
    ap_class2 = result["per_class"][2]["ap"]
    assert abs(ap_class1 - expected_ap) < 1e-6, f"Clase 1: AP esperado {expected_ap}, obtenido {ap_class1}"
    assert abs(ap_class2 - 1.0) < 1e-9, f"Clase 2: AP esperado 1.0, obtenido {ap_class2}"
    for c in range(3, 10):
        assert result["per_class"][c]["ap"] == 0.0, f"Clase {c} sin GT debe dar AP=0.0"

    # Igual en todos los umbrales para ambas clases (el match es ~perfecto o nulo)
    for thr in IOU_THRESHOLDS:
        thr_ap1 = result["per_class"][1]["per_threshold"][thr]
        thr_ap2 = result["per_class"][2]["per_threshold"][thr]
        assert abs(thr_ap1 - expected_ap) < 1e-6, f"Umbral {thr} clase1: {thr_ap1} != {expected_ap}"
        assert abs(thr_ap2 - 1.0) < 1e-9, f"Umbral {thr} clase2: {thr_ap2} != 1.0"

    expected_macro = (expected_ap + 1.0 + 0.0 * 7) / 9.0
    assert abs(result["macro_ap"] - expected_macro) < 1e-6, (
        f"Macro AP esperado {expected_macro}, obtenido {result['macro_ap']}"
    )

    # 4) parse_target: formatos válidos, 'none' y tokens corruptos.
    assert parse_target("none", has_score=False) == []
    assert parse_target("NONE", has_score=True) == []
    assert parse_target("", has_score=False) == []
    gt_dets = parse_target("1 100.0 100.0 50.0 30.0 0.0", has_score=False)
    assert len(gt_dets) == 1 and gt_dets[0]["category_id"] == 1 and gt_dets[0]["score"] == 1.0
    pred_dets = parse_target("0.9 1 100.0 100.0 50.0 30.0 0.0;0.5 2 1 1 1 1 1", has_score=True)
    assert len(pred_dets) == 2 and pred_dets[0]["score"] == 0.9 and pred_dets[1]["category_id"] == 2
    assert parse_target("token corrupto", has_score=True) == []  # longitud incorrecta -> descartado

    print("OK — caso sintetico de control PASO.")
    print(f"  rotated_iou(identico)=1.0, rotated_iou(lejano)=0.0, rotated_iou(desplazado)={iou_shifted:.6f} (esperado 0.333333)")
    print(f"  compute_ap manual    = {ap_manual:.6f} (esperado {expected_ap:.6f})")
    print(f"  AP clase 1 (TP/FN/FP)= {ap_class1:.6f} (esperado {expected_ap:.6f})")
    print(f"  AP clase 2 (perfecto)= {ap_class2:.6f} (esperado 1.0)")
    print(f"  Macro AP-rIoU        = {result['macro_ap']:.6f} (esperado {expected_macro:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalua Macro AP-rIoU @ [0.50:0.80] (formato submission)")
    parser.add_argument("--gt", type=Path, help="CSV (Id,Target) con ground truth: 'category_id cx cy w h angle'")
    parser.add_argument("--preds", type=Path, help="CSV (Id,Target) en formato submission: 'score category_id cx cy w h angle'")
    parser.add_argument("--ids", type=Path, help="Archivo .txt con frame_ids a evaluar, uno por linea")
    parser.add_argument("--selftest", action="store_true", help="Ejecuta el caso sintetico de control y termina")
    args = parser.parse_args()

    if args.selftest or not (args.gt and args.preds and args.ids):
        if not args.selftest:
            log.info("No se especificaron --gt/--preds/--ids; ejecutando caso sintetico de control.")
        _run_self_test()
    else:
        gt_data = load_csv(args.gt, has_score=False)
        preds_data = load_csv(args.preds, has_score=True)
        eval_frame_ids = args.ids.read_text().strip().splitlines()

        result = evaluate(gt_data, preds_data, eval_frame_ids)

        print(f"\nMacro AP-rIoU @ [0.50:0.80] = {result['macro_ap']:.4f}")
        for cat in CLASS_IDS:
            print(f"  Clase {cat}: AP={result['per_class'][cat]['ap']:.4f}")
