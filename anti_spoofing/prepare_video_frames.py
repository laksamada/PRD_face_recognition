from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.face_engine import get_engine

DEFAULT_VIDEO_MANIFEST = ROOT / "data" / "anti_spoof_video_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "anti_spoof_frames"
DEFAULT_FRAME_MANIFEST = ROOT / "data" / "anti_spoof_frame_manifest.csv"
DEFAULT_SUMMARY_JSON = ROOT / "data" / "anti_spoof_frame_summary.json"
DISPLAY_ATTACK_TYPES = {"screen_replay", "display_spoof"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ekstrak crop wajah dari video anti-spoofing.")
    parser.add_argument("--video-manifest", type=Path, default=DEFAULT_VIDEO_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-manifest", type=Path, default=DEFAULT_FRAME_MANIFEST)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--frames-per-video", type=int, default=6)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-side-for-detect", type=int, default=1280)
    parser.add_argument("--margin-ratio", type=float, default=0.18)
    parser.add_argument("--val-fraction-dataset1", type=float, default=0.20)
    parser.add_argument("--val-fraction-dataset2", type=float, default=0.10)
    parser.add_argument("--val-fraction-dataset3", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos", type=int, default=0, help="0 berarti semua video.")
    parser.add_argument("--display-frames-per-video", type=int, default=10)
    parser.add_argument("--display-candidate-multiplier", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def assign_model_splits(
    rows: list[dict],
    val_fraction_d1: float,
    val_fraction_d2: float,
    val_fraction_d3: float,
) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    result: list[dict] = []

    for row in rows:
        row = dict(row)
        source_dataset = row["source_dataset"]
        split = row["split"]
        label = row["label"]

        if source_dataset == "dataset1" and split == "test":
            row["model_split"] = "test"
            result.append(row)
        else:
            grouped[(source_dataset, split, label)].append(row)

    for (source_dataset, split, label), group_rows in grouped.items():
        group_rows = sorted(group_rows, key=lambda item: item["relative_path"])
        if source_dataset == "dataset1":
            val_fraction = val_fraction_d1
        elif source_dataset == "dataset3":
            val_fraction = val_fraction_d3
        else:
            val_fraction = val_fraction_d2

        val_count = int(round(len(group_rows) * val_fraction))
        if len(group_rows) > 1:
            val_count = max(1, min(len(group_rows) - 1, val_count))
        else:
            val_count = 0

        val_indices = set(np.linspace(0, len(group_rows) - 1, num=val_count, dtype=int).tolist()) if val_count > 0 else set()
        for idx, row in enumerate(group_rows):
            row = dict(row)
            row["model_split"] = "val" if idx in val_indices else "train"
            result.append(row)

    return sorted(result, key=lambda item: (item["source_dataset"], item["relative_path"]))


def parse_int(value: str | int | float) -> int:
    return int(float(value))


def sample_frame_indices(frame_count: int, target_count: int) -> list[int]:
    if frame_count <= 0:
        return []

    start = max(0, int(frame_count * 0.12))
    end = min(frame_count - 1, max(start, int(frame_count * 0.88)))
    count = max(target_count, 1)
    return sorted(set(np.linspace(start, end, num=count, dtype=int).tolist()))


def resize_for_detection(frame: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = 1.0
    max_current = max(height, width)
    if max_current > max_side:
        scale = max_side / max_current
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    return frame, scale


def largest_face(faces) -> object | None:
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
    )


def crop_face(frame: np.ndarray, bbox: np.ndarray, margin_ratio: float) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox.astype(int)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(frame.shape[1], x2 + margin_x)
    y2 = min(frame.shape[0], y2 + margin_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def extract_frames_for_video(
    row: dict,
    output_dir: Path,
    engine,
    frames_per_video: int,
    candidate_multiplier: int,
    image_size: int,
    max_side_for_detect: int,
    margin_ratio: float,
    overwrite: bool,
) -> tuple[list[dict], Counter]:
    counters = Counter()
    full_path = Path(row["full_path"])
    video_key = f'{row["source_dataset"]}__{Path(row["relative_path"]).with_suffix("").as_posix().replace("/", "__")}'
    target_dir = output_dir / row["model_split"] / row["label"] / video_key
    target_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(full_path))
    if not cap.isOpened():
        counters["video_open_failed"] += 1
        return [], counters

    frame_count = parse_int(row["frame_count"])
    local_frames_per_video = frames_per_video
    local_candidate_multiplier = candidate_multiplier
    if row.get("attack_type") in DISPLAY_ATTACK_TYPES:
        local_frames_per_video = max(local_frames_per_video, int(row.get("display_frames_per_video", local_frames_per_video)))
        local_candidate_multiplier = max(local_candidate_multiplier, int(row.get("display_candidate_multiplier", local_candidate_multiplier)))

    candidate_count = max(local_frames_per_video * local_candidate_multiplier, local_frames_per_video)
    frame_indices = sample_frame_indices(frame_count, candidate_count)

    extracted_rows: list[dict] = []
    saved = 0

    try:
        for frame_index in frame_indices:
            if saved >= local_frames_per_video:
                break

            output_path = target_dir / f"frame_{frame_index:05d}.jpg"
            if output_path.exists() and not overwrite:
                extracted_rows.append(
                    {
                        "image_path": str(output_path),
                        "relative_image_path": output_path.relative_to(ROOT).as_posix(),
                        "label": row["label"],
                        "label_id": 1 if row["label"] == "real" else 0,
                        "model_split": row["model_split"],
                        "source_dataset": row["source_dataset"],
                        "attack_type": row["attack_type"],
                        "source_group": row.get("source_group", ""),
                        "video_path": row["full_path"],
                        "relative_video_path": row["relative_path"],
                        "frame_index": frame_index,
                        "video_fps": row["fps"],
                        "video_frame_count": row["frame_count"],
                        "width": row["width"],
                        "height": row["height"],
                    }
                )
                saved += 1
                counters["existing_crop_reused"] += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                counters["frame_read_failed"] += 1
                continue

            detect_frame, _ = resize_for_detection(frame, max_side_for_detect)
            faces = engine.get(detect_frame)
            face = largest_face(faces)
            if face is None:
                counters["face_not_found"] += 1
                continue

            crop = crop_face(detect_frame, face.bbox, margin_ratio)
            if crop is None:
                counters["crop_failed"] += 1
                continue

            crop = cv2.resize(crop, (image_size, image_size))
            if not cv2.imwrite(str(output_path), crop):
                counters["image_write_failed"] += 1
                continue

            extracted_rows.append(
                {
                    "image_path": str(output_path),
                    "relative_image_path": output_path.relative_to(ROOT).as_posix(),
                    "label": row["label"],
                    "label_id": 1 if row["label"] == "real" else 0,
                    "model_split": row["model_split"],
                    "source_dataset": row["source_dataset"],
                    "attack_type": row["attack_type"],
                    "source_group": row.get("source_group", ""),
                    "video_path": row["full_path"],
                    "relative_video_path": row["relative_path"],
                    "frame_index": frame_index,
                    "video_fps": row["fps"],
                    "video_frame_count": row["frame_count"],
                    "width": row["width"],
                    "height": row["height"],
                }
            )
            saved += 1
            counters["crop_saved"] += 1
    finally:
        cap.release()

    if saved == 0:
        counters["video_without_any_crop"] += 1

    return extracted_rows, counters


def write_frame_manifest(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "relative_image_path",
        "label",
        "label_id",
        "model_split",
        "source_dataset",
        "attack_type",
        "source_group",
        "video_path",
        "relative_video_path",
        "frame_index",
        "video_fps",
        "video_frame_count",
        "width",
        "height",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(rows: list[dict], counters: Counter, video_rows: list[dict]) -> dict:
    by_split = Counter()
    by_label = Counter()
    by_attack = Counter()
    videos_by_split = Counter()

    for row in rows:
        by_split[row["model_split"]] += 1
        by_label[row["label"]] += 1
        by_attack[row["attack_type"]] += 1

    for row in video_rows:
        videos_by_split[row["model_split"]] += 1

    return {
        "total_videos_considered": len(video_rows),
        "total_frame_crops": len(rows),
        "frames_by_split": dict(by_split),
        "frames_by_label": dict(by_label),
        "frames_by_attack_type": dict(by_attack),
        "videos_by_split": dict(videos_by_split),
        "counters": dict(counters),
    }


def main() -> int:
    args = parse_args()
    args.video_manifest = args.video_manifest if args.video_manifest.is_absolute() else (ROOT / args.video_manifest).resolve()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    args.frame_manifest = args.frame_manifest if args.frame_manifest.is_absolute() else (ROOT / args.frame_manifest).resolve()
    args.summary_json = args.summary_json if args.summary_json.is_absolute() else (ROOT / args.summary_json).resolve()

    video_rows = read_rows(args.video_manifest)
    video_rows = [row for row in video_rows if str(row.get("opened", "True")).lower() == "true"]
    video_rows = assign_model_splits(
        video_rows,
        val_fraction_d1=args.val_fraction_dataset1,
        val_fraction_d2=args.val_fraction_dataset2,
        val_fraction_d3=args.val_fraction_dataset3,
    )

    for row in video_rows:
        row["display_frames_per_video"] = args.display_frames_per_video
        row["display_candidate_multiplier"] = args.display_candidate_multiplier

    if args.max_videos > 0:
        video_rows = video_rows[: args.max_videos]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    all_rows: list[dict] = []
    counters = Counter()

    total = len(video_rows)
    for idx, row in enumerate(video_rows, start=1):
        extracted_rows, local_counters = extract_frames_for_video(
            row=row,
            output_dir=args.output_dir,
            engine=engine,
            frames_per_video=args.frames_per_video,
            candidate_multiplier=args.candidate_multiplier,
            image_size=args.image_size,
            max_side_for_detect=args.max_side_for_detect,
            margin_ratio=args.margin_ratio,
            overwrite=args.overwrite,
        )
        all_rows.extend(extracted_rows)
        counters.update(local_counters)

        if idx % 10 == 0 or idx == total:
            print(
                f"[{idx:>3}/{total}] {row['relative_path']} -> "
                f"{len(extracted_rows)} crop(s), total={len(all_rows)}"
            )

    write_frame_manifest(all_rows, args.frame_manifest)
    summary = build_summary(all_rows, counters, video_rows)
    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nFrame manifest ditulis ke: {args.frame_manifest}")
    print(f"Ringkasan ditulis ke: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
