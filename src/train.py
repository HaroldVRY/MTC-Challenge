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
Entrena YOLO-OBB (YOLOv8/v11) sobre el dataset de la competencia MTC.

Uso:
    python -m src.train --config configs/train.yaml --data configs/dataset.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    """Fija semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(train_cfg: dict, data_cfg_path: Path) -> None:
    set_seeds(42)

    model_name: str = train_cfg.get("model", "yolov8m-obb.pt")
    log.info("Cargando modelo: %s", model_name)
    model = YOLO(model_name)

    device = train_cfg.get("device", "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        log.warning("GPU no disponible, usando CPU (entrenamiento muy lento)")
        device = "cpu"

    log.info("Iniciando entrenamiento en device=%s", device)
    results = model.train(
        data=str(data_cfg_path),
        epochs=train_cfg.get("epochs", 100),
        imgsz=train_cfg.get("imgsz", 1280),
        batch=train_cfg.get("batch", 8),
        device=device,
        workers=train_cfg.get("workers", 4),
        patience=train_cfg.get("patience", 20),
        optimizer=train_cfg.get("optimizer", "AdamW"),
        lr0=train_cfg.get("lr0", 0.001),
        lrf=train_cfg.get("lrf", 0.01),
        momentum=train_cfg.get("momentum", 0.937),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        warmup_epochs=train_cfg.get("warmup_epochs", 3),
        hsv_h=train_cfg.get("hsv_h", 0.015),
        hsv_s=train_cfg.get("hsv_s", 0.7),
        hsv_v=train_cfg.get("hsv_v", 0.4),
        degrees=train_cfg.get("degrees", 10.0),
        translate=train_cfg.get("translate", 0.1),
        scale=train_cfg.get("scale", 0.5),
        fliplr=train_cfg.get("fliplr", 0.5),
        flipud=train_cfg.get("flipud", 0.0),
        mosaic=train_cfg.get("mosaic", 1.0),
        mixup=train_cfg.get("mixup", 0.1),
        copy_paste=train_cfg.get("copy_paste", 0.1),
        project="runs",
        name="mtc_obb",
        save=True,
        save_period=train_cfg.get("save_period", 10),
    )
    log.info("Entrenamiento finalizado. Resultados en runs/mtc_obb/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena YOLO-OBB")
    parser.add_argument("--config", default=Path("configs/train.yaml"), type=Path)
    parser.add_argument("--data", default=Path("configs/data.yaml"), type=Path)
    args = parser.parse_args()

    train_cfg = load_config(args.config)
    train(train_cfg, args.data)
