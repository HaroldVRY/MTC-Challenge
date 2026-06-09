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
Genera el split train/val agrupado por video_id y construye la estructura YOLO.

Algoritmo (dos fases):
  Fase 1 — greedy: para cada clase (de más escasa a más frecuente), agrega videos
    a val hasta cumplir min_instances. Articulado solo tiene 5 videos; 1 es suficiente.
  Fase 2 — relleno aleatorio: agrega videos al azar hasta alcanzar ~val_ratio de frames.

Salida:
  data/yolo/images/{train,val}/  ← hard-links a data/raw/train_frames/
  data/yolo/labels/{train,val}/  ← hard-links a data/yolo/labels/
  configs/data.yaml

Uso:
    python -m src.make_split \
        --csv train.csv \
        --frames data/raw/train_frames \
        --labels data/yolo/labels \
        --yolo-dir data/yolo \
        --cfg-dir configs \
        --val-ratio 0.15 \
        --min-instances 15 \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = [
    "auto", "combi", "microbus", "minibus", "omnibus",
    "articulado", "camion", "mototaxi", "motocicleta",
]
NUM_CLASSES = 9


# ── Estadísticas por video ─────────────────────────────────────────────────

def compute_video_stats(
    df: pd.DataFrame,
) -> tuple[dict[str, Counter], dict[str, int]]:
    """Devuelve (video_class_counts, video_frame_counts).

    video_class_counts[video_id][category_id(1-9)] = n instancias
    video_frame_counts[video_id]                   = n frames
    """
    vid_cls: dict[str, Counter] = defaultdict(Counter)
    vid_frm: dict[str, int] = Counter()

    for row in df.itertuples(index=False):
        vid = row.Id.rsplit("_", 1)[0]
        vid_frm[vid] += 1
        t = str(row.Target).strip()
        if t.lower() == "none":
            continue
        for tok in t.split(";"):
            parts = tok.strip().split()
            if len(parts) == 6:
                vid_cls[vid][int(parts[0])] += 1

    return dict(vid_cls), dict(vid_frm)


# ── Split greedy + relleno aleatorio ──────────────────────────────────────

def greedy_split(
    vid_cls: dict[str, Counter],
    vid_frm: dict[str, int],
    all_videos: list[str],
    val_ratio: float = 0.15,
    min_instances: int = 15,
    seed: int = 42,
) -> tuple[set[str], set[str], list[str]]:
    """Devuelve (train_videos, val_videos, warnings).

    Garantiza que val tenga >= min_instances de cada clase (o todo lo disponible
    si la clase es tan escasa que no alcanza).
    """
    rng = random.Random(seed)
    warnings: list[str] = []

    # Total de instancias por clase
    total_cls: Counter = Counter()
    for counts in vid_cls.values():
        total_cls.update(counts)

    val_set: set[str] = set()
    val_cls: Counter = Counter()

    # ── Fase 1: greedy (clase más escasa primero) ─────────────────────────
    classes_by_rarity = sorted(
        range(1, NUM_CLASSES + 1),
        key=lambda c: total_cls.get(c, 0),
    )

    for cls in classes_by_rarity:
        total = total_cls.get(cls, 0)
        if total == 0:
            warnings.append(
                f"WARN  clase {cls} ({CLASS_NAMES[cls-1]}): "
                "sin instancias en el dataset."
            )
            continue

        cls_min = min(min_instances, total)
        if cls_min < min_instances:
            warnings.append(
                f"INFO  clase {cls} ({CLASS_NAMES[cls-1]}): solo {total} instancias "
                f"en todo el dataset; meta ajustada a {cls_min}."
            )

        if val_cls[cls] >= cls_min:
            continue  # ya satisfecho por videos añadidos antes

        # Candidatos: videos con esta clase, no en val aún, ordenados desc
        candidates = sorted(
            (
                (vid, vid_cls.get(vid, Counter()).get(cls, 0))
                for vid in all_videos
                if vid not in val_set and vid_cls.get(vid, Counter()).get(cls, 0) > 0
            ),
            key=lambda x: -x[1],
        )

        for vid, n in candidates:
            if val_cls[cls] >= cls_min:
                break
            val_set.add(vid)
            for c, cnt in vid_cls.get(vid, {}).items():
                val_cls[c] += cnt

        # Verificar si se alcanzó el mínimo
        if val_cls[cls] < cls_min:
            warnings.append(
                f"WARN  clase {cls} ({CLASS_NAMES[cls-1]}): "
                f"solo se pudieron colocar {val_cls[cls]} instancias en val "
                f"(meta: {cls_min}). Considera K-fold."
            )

    # ── Fase 2: relleno aleatorio hasta val_ratio ─────────────────────────
    total_frames = sum(vid_frm.values())
    target_val_frames = int(total_frames * val_ratio)
    current_val_frames = sum(vid_frm.get(v, 0) for v in val_set)

    remaining = [v for v in all_videos if v not in val_set]
    rng.shuffle(remaining)

    for vid in remaining:
        if current_val_frames >= target_val_frames:
            break
        val_set.add(vid)
        current_val_frames += vid_frm.get(vid, 0)

    train_set = set(all_videos) - val_set
    return train_set, val_set, warnings


# ── Creación de hard-links / symlinks ──────────────────────────────────────

def _probe_link_method(probe_dir: Path) -> str:
    """Detecta el mejor método de enlace disponible: 'hard', 'sym' o 'txt'."""
    src = probe_dir / ".make_split_probe_src"
    dst = probe_dir / ".make_split_probe_dst"
    src.write_text("")
    try:
        os.link(str(src), str(dst))
        dst.unlink(missing_ok=True)
        src.unlink(missing_ok=True)
        return "hard"
    except OSError:
        pass
    try:
        os.symlink(str(src.resolve()), str(dst))
        dst.unlink(missing_ok=True)
        src.unlink(missing_ok=True)
        return "sym"
    except OSError:
        pass
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
    return "txt"


def create_links(
    frame_ids: list[str],
    split: str,
    frames_src: Path,
    labels_src: Path,
    yolo_dir: Path,
    method: str,
) -> tuple[int, int, int]:
    """Crea images/{split}/ y labels/{split}/ con enlaces.

    Returns (n_creados, n_existían, n_faltaban_img).
    """
    img_dst = yolo_dir / "images" / split
    lbl_dst = yolo_dir / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_miss = 0

    def link(src: Path, dst: Path) -> None:
        if method == "hard":
            os.link(str(src), str(dst))
        else:  # sym
            os.symlink(str(src.resolve()), str(dst))

    for fid in tqdm(frame_ids, desc=f"  {split:5s}", ncols=80, leave=True):
        img_src = frames_src / f"{fid}.jpg"
        img_lnk = img_dst / f"{fid}.jpg"
        lbl_lnk = lbl_dst / f"{fid}.txt"
        lbl_s   = labels_src / f"{fid}.txt"

        if img_lnk.exists():
            n_skip += 1
        elif img_src.exists():
            link(img_src, img_lnk)
            n_ok += 1
        else:
            n_miss += 1
            continue  # no imagen → no label

        if not lbl_lnk.exists() and lbl_s.exists():
            link(lbl_s, lbl_lnk)

    return n_ok, n_skip, n_miss


def write_txt_lists(
    train_ids: list[str],
    val_ids: list[str],
    frames_src: Path,
    yolo_dir: Path,
) -> None:
    """Fallback cuando no se pueden crear links: escribe train.txt / val.txt."""
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        lines = [str((frames_src / f"{fid}.jpg").resolve()) for fid in ids]
        (yolo_dir / f"{split}.txt").write_text("\n".join(lines) + "\n")
    log.info("Escritos train.txt y val.txt en %s", yolo_dir)


# ── configs/data.yaml ──────────────────────────────────────────────────────

def write_data_yaml(
    yolo_dir: Path,
    cfg_dir: Path,
    use_txt: bool = False,
) -> Path:
    yolo_abs = str(yolo_dir.resolve()).replace("\\", "/")
    data = {
        "path":  yolo_abs,
        "train": "train.txt" if use_txt else "images/train",
        "val":   "val.txt"   if use_txt else "images/val",
        "nc":    NUM_CLASSES,
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out = cfg_dir / "data.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return out


# ── Reporte ────────────────────────────────────────────────────────────────

def report(
    train_videos: set[str],
    val_videos: set[str],
    train_ids: list[str],
    val_ids: list[str],
    vid_cls: dict[str, Counter],
) -> None:
    n_tr_v = len(train_videos)
    n_vl_v = len(val_videos)
    pct_vl = 100 * n_vl_v / (n_tr_v + n_vl_v)

    def agg(vids: set[str]) -> Counter:
        c: Counter = Counter()
        for vid in vids:
            c.update(vid_cls.get(vid, {}))
        return c

    tr_cls = agg(train_videos)
    vl_cls = agg(val_videos)

    print("\n" + "=" * 66)
    print("RESUMEN DEL SPLIT")
    print("=" * 66)
    print(f"  Videos  train : {n_tr_v:>5}  ({100 - pct_vl:.1f}%)")
    print(f"  Videos  val   : {n_vl_v:>5}  ({pct_vl:.1f}%)")
    print(f"  Frames  train : {len(train_ids):>7,}")
    print(f"  Frames  val   : {len(val_ids):>7,}")
    print()
    print(f"  {'Clase':<13} {'Train':>9} {'Val':>8} {'Total':>9} {'Val%':>6}")
    print("  " + "-" * 49)
    for cls in range(1, NUM_CLASSES + 1):
        tr  = tr_cls.get(cls, 0)
        vl  = vl_cls.get(cls, 0)
        tot = tr + vl
        pct = 100 * vl / tot if tot else 0.0
        flag = "  <<< SOLO 5 VIDEOS" if CLASS_NAMES[cls - 1] == "articulado" else ""
        print(f"  {CLASS_NAMES[cls-1]:<13} {tr:>9,} {vl:>8,} {tot:>9,} {pct:>5.1f}%{flag}")
    print("  " + "-" * 49)
    tot_tr = sum(tr_cls.values())
    tot_vl = sum(vl_cls.values())
    tot    = tot_tr + tot_vl
    print(f"  {'TOTAL':<13} {tot_tr:>9,} {tot_vl:>8,} {tot:>9,} {100*tot_vl/tot:>5.1f}%")
    print("=" * 66)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split train/val agrupado por video + estructura YOLO"
    )
    parser.add_argument("--csv",           default=Path("train.csv"),              type=Path)
    parser.add_argument("--frames",        default=Path("data/raw/train_frames"),  type=Path)
    parser.add_argument("--labels",        default=Path("data/yolo/labels"),       type=Path)
    parser.add_argument("--yolo-dir",      default=Path("data/yolo"),              type=Path)
    parser.add_argument("--cfg-dir",       default=Path("configs"),                type=Path)
    parser.add_argument("--val-ratio",     default=0.15, type=float)
    parser.add_argument("--min-instances", default=15,   type=int,
                        help="Instancias mínimas de cada clase en val")
    parser.add_argument("--seed",          default=42,   type=int)
    parser.add_argument("--dry-run",       action="store_true",
                        help="Solo calcula el split, no crea archivos")
    args = parser.parse_args()

    # ── Cargar CSV ────────────────────────────────────────────────────────
    log.info("Cargando %s ...", args.csv)
    df = pd.read_csv(args.csv)
    id_col  = next(c for c in df.columns if c.lower() == "id")
    tgt_col = next(c for c in df.columns if c.lower() == "target")
    df = df.rename(columns={id_col: "Id", tgt_col: "Target"})

    vid_cls, vid_frm = compute_video_stats(df)
    all_videos = sorted(vid_frm.keys())
    log.info("Videos: %d  |  Frames: %d", len(all_videos), sum(vid_frm.values()))

    # ── Calcular split ────────────────────────────────────────────────────
    log.info("Calculando split (val_ratio=%.2f, min_instances=%d, seed=%d) ...",
             args.val_ratio, args.min_instances, args.seed)
    train_videos, val_videos, warnings = greedy_split(
        vid_cls, vid_frm, all_videos,
        val_ratio=args.val_ratio,
        min_instances=args.min_instances,
        seed=args.seed,
    )

    if warnings:
        print("\nAVISOS DEL SPLIT:")
        for w in warnings:
            print(" ", w)

    # ── Frame IDs por split ───────────────────────────────────────────────
    df["video_id"] = df["Id"].str.rsplit("_", n=1).str[0]
    train_ids = df[df["video_id"].isin(train_videos)]["Id"].tolist()
    val_ids   = df[df["video_id"].isin(val_videos)]["Id"].tolist()

    report(train_videos, val_videos, train_ids, val_ids, vid_cls)

    if args.dry_run:
        log.info("--dry-run activado; no se crean archivos.")
        return

    # ── Detectar método de enlace ─────────────────────────────────────────
    method = _probe_link_method(args.yolo_dir)
    log.info("Método de enlace detectado: %s", method)

    if method == "txt":
        log.warning(
            "No se pueden crear hard-links ni symlinks. "
            "Generando train.txt / val.txt como fallback."
        )
        write_txt_lists(train_ids, val_ids, args.frames, args.yolo_dir)
        yaml_path = write_data_yaml(args.yolo_dir, args.cfg_dir, use_txt=True)
    else:
        log.info("Creando estructura YOLO con %s ...", method)
        for split, ids in [("train", train_ids), ("val", val_ids)]:
            n_ok, n_skip, n_miss = create_links(
                ids, split, args.frames, args.labels, args.yolo_dir, method
            )
            log.info(
                "  %s: %d nuevos, %d ya existían, %d imgs no encontradas",
                split, n_ok, n_skip, n_miss,
            )
        yaml_path = write_data_yaml(args.yolo_dir, args.cfg_dir, use_txt=False)

    log.info("data.yaml escrito en %s", yaml_path)

    # ── Guardar listas de IDs para referencia ─────────────────────────────
    (args.yolo_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n")
    (args.yolo_dir / "val_ids.txt").write_text("\n".join(val_ids) + "\n")
    log.info("IDs guardados en %s/{train,val}_ids.txt", args.yolo_dir)


if __name__ == "__main__":
    main()
