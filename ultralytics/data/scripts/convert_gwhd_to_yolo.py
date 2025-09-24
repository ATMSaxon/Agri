#!/usr/bin/env python3
"""
Convert GWHD 2021 dataset to Ultralytics YOLO format.

Input directory structure (as provided):
  datasets/gwhd_2021/
    - competition_train.csv
    - competition_val.csv
    - competition_test.csv (optional or empty boxes)
    - images/*.png

Output directory structure (under --out-dir):
  gwhd_2021_yolo/
    images/{train,val,test}/<image>.png -> symlink to original
    labels/{train,val,test}/<image>.txt  (YOLO format labels)
    data.yaml

CSV schema:
  image_name, BoxesString, domain
Where BoxesString is a semicolon-separated list of "x_min y_min x_max y_max".

All labels use a single class 0 (wheat head).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


@dataclass
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_yolo(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        # Convert to YOLO (x_center, y_center, width, height), normalized [0,1]
        x_center = (self.x_min + self.x_max) / 2.0
        y_center = (self.y_min + self.y_max) / 2.0
        w = max(0.0, self.x_max - self.x_min)
        h = max(0.0, self.y_max - self.y_min)
        return (
            x_center / float(img_w),
            y_center / float(img_h),
            w / float(img_w),
            h / float(img_h),
        )


def parse_boxes_string(s: str) -> List[Box]:
    s = (s or "").strip()
    if not s:
        return []
    boxes: List[Box] = []
    for token in s.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split()
        if len(parts) != 4:
            # Skip malformed entries
            continue
        try:
            x1, y1, x2, y2 = map(float, parts)
        except Exception:
            continue
        # Ensure proper ordering
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
        boxes.append(Box(x_min, y_min, x_max, y_max))
    return boxes


def read_split_csv(csv_path: Path) -> Dict[str, List[Box]]:
    mapping: Dict[str, List[Box]] = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        # Normalize column names
        field_map = {k.lower(): k for k in reader.fieldnames or []}
        image_key = field_map.get("image_name") or field_map.get("image") or "image_name"
        boxes_key = field_map.get("boxesstring") or field_map.get("boxes") or "boxesstring"
        for row in reader:
            img_name = (row.get(image_key) or "").strip()
            boxes_raw = row.get(boxes_key) or ""
            boxes = parse_boxes_string(boxes_raw)
            mapping[img_name] = boxes
    return mapping


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_yolo_labels_for_split(
    split_name: str,
    img_to_boxes: Dict[str, List[Box]],
    images_dir: Path,
    out_root: Path,
    symlink: bool = True,
) -> Tuple[int, int]:
    images_out = out_root / "images" / split_name
    labels_out = out_root / "labels" / split_name
    ensure_dir(images_out)
    ensure_dir(labels_out)

    num_images = 0
    num_boxes = 0

    for img_name, boxes in img_to_boxes.items():
        img_src = images_dir / img_name
        if not img_src.exists():
            # Some CSV rows might reference missing images
            continue

        # Determine image size
        try:
            with Image.open(img_src) as im:
                img_w, img_h = im.size
        except Exception:
            continue

        # Link/copy image
        img_dst = images_out / img_name
        try:
            if symlink:
                if img_dst.exists() or img_dst.is_symlink():
                    img_dst.unlink()
                os.symlink(os.fspath(img_src), os.fspath(img_dst))
            else:
                if not img_dst.exists():
                    # copy without importing shutil lazily
                    data = img_src.read_bytes()
                    img_dst.write_bytes(data)
        except FileExistsError:
            pass

        # Write labels
        label_dst = labels_out / (Path(img_name).with_suffix(".txt").name)
        lines: List[str] = []
        for b in boxes:
            x, y, w, h = b.to_yolo(img_w, img_h)
            # Clip to [0,1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            if w <= 0.0 or h <= 0.0:
                continue
            lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        label_dst.write_text("\n".join(lines) + ("\n" if lines else ""))

        num_images += 1
        num_boxes += len(lines)

    return num_images, num_boxes


def write_data_yaml(out_root: Path, dataset_root: Path, include_test: bool) -> None:
    yaml_path = out_root / "data.yaml"
    train_images = out_root / "images" / "train"
    val_images = out_root / "images" / "val"
    test_images = out_root / "images" / "test"

    content_lines = [
        f"path: {dataset_root}",
        f"train: {train_images}",
        f"val: {val_images}",
    ]
    if include_test:
        content_lines.append(f"test: {test_images}")
    content_lines += [
        "names:",
        "  0: wheat_head",
    ]
    yaml_path.write_text("\n".join(content_lines) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert GWHD 2021 to YOLO format")
    parser.add_argument("--src", type=str, default=str(Path.cwd() / "datasets" / "gwhd_2021"), help="Source GWHD directory")
    parser.add_argument("--out-dir", type=str, default=str(Path.cwd() / "datasets" / "gwhd_2021_yolo"), help="Output directory for YOLO dataset")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    args = parser.parse_args(list(argv) if argv is not None else None)

    src_root = Path(args.src).resolve()
    out_root = Path(args.out_dir).resolve()
    ensure_dir(out_root)

    images_dir = src_root / "images"
    train_csv = src_root / "competition_train.csv"
    val_csv = src_root / "competition_val.csv"
    test_csv = src_root / "competition_test.csv"

    if not images_dir.exists():
        print(f"ERROR: images directory not found: {images_dir}", file=sys.stderr)
        return 1

    train_map = read_split_csv(train_csv)
    val_map = read_split_csv(val_csv)
    test_map = read_split_csv(test_csv) if test_csv.exists() else {}

    # Write splits
    print("Writing train split...")
    n_img_train, n_box_train = write_yolo_labels_for_split("train", train_map, images_dir, out_root, symlink=(not args.copy))
    print(f"Train: {n_img_train} images, {n_box_train} boxes")

    print("Writing val split...")
    n_img_val, n_box_val = write_yolo_labels_for_split("val", val_map, images_dir, out_root, symlink=(not args.copy))
    print(f"Val: {n_img_val} images, {n_box_val} boxes")

    include_test = len(test_map) > 0
    if include_test:
        print("Writing test split...")
        n_img_test, n_box_test = write_yolo_labels_for_split("test", test_map, images_dir, out_root, symlink=(not args.copy))
        print(f"Test: {n_img_test} images, {n_box_test} boxes")

    # Write data.yaml
    write_data_yaml(out_root, out_root, include_test=include_test)
    print(f"Done. YOLO dataset at: {out_root}")
    print(f"data.yaml: {out_root / 'data.yaml'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





