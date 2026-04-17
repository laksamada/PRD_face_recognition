from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent
DATASET1_DIR = ROOT / "dataset1"
DATASET2_DIR = ROOT / "dataset2"
DATASET3_DIR = ROOT / "dataset3"
OUTPUT_CSV = ROOT / "data" / "anti_spoof_video_manifest.csv"
OUTPUT_JSON = ROOT / "data" / "anti_spoof_video_summary.json"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

DATASET2_LABELS: dict[str, tuple[str | None, str]] = {
    "3D_paper_mask": ("fake", "3d_paper_mask"),
    "Cutout_attacks": ("fake", "cutout_attack"),
    "Latex_mask": ("fake", "latex_mask"),
    "Replay_display_attacks": (None, "mixed_replay_display"),
    "Replay_mobile_attacks": ("fake", "mobile_replay"),
    "Selfies": (None, "selfie_images_only"),
    "Silicone_mask": ("fake", "silicone_mask"),
    "Textile 3D Face Mask Attack Sample": ("fake", "textile_3d_mask"),
    "Wrapped_3D_paper_mask": ("fake", "wrapped_3d_paper_mask"),
}


def video_metadata(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        return {
            "opened": cap.isOpened(),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        cap.release()


def rows_from_dataset1() -> list[dict]:
    csv_path = DATASET1_DIR / "real_and_fake.csv"
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            rel_path = Path(item["file"])
            full_path = DATASET1_DIR / rel_path
            if full_path.suffix.lower() not in VIDEO_EXTS:
                continue

            meta = video_metadata(full_path)
            rows.append(
                {
                    "source_dataset": "dataset1",
                    "relative_path": rel_path.as_posix(),
                    "full_path": str(full_path),
                    "split": item["split"],
                    "label": "real" if item["type"] == "real" else "fake",
                    "attack_type": "none" if item["type"] == "real" else "generic_attack",
                    **meta,
                }
            )
    return rows


def label_dataset2_video(path: Path) -> tuple[str | None, str, str]:
    rel = path.relative_to(DATASET2_DIR)
    top = rel.parts[0]
    label, attack_type = DATASET2_LABELS.get(top, (None, "unknown"))

    if top == "Replay_display_attacks":
        second = rel.parts[1] if len(rel.parts) > 1 else ""
        if second.lower() == "real":
            return "real", "display_real", "replay_display_real"
        if second.lower() == "screen":
            return "fake", "screen_replay", "replay_display_screen"
        return None, attack_type, "replay_display_unknown"

    if label is None:
        return None, attack_type, top.lower().replace(" ", "_")
    return label, attack_type, top.lower().replace(" ", "_")


def rows_from_dataset2() -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    ignored = Counter()

    for path in DATASET2_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTS:
            ignored["non_video"] += 1
            continue

        label, attack_type, source_group = label_dataset2_video(path)
        if label is None:
            ignored[f"unmapped:{source_group}"] += 1
            continue

        meta = video_metadata(path)
        rows.append(
            {
                "source_dataset": "dataset2",
                "relative_path": path.relative_to(DATASET2_DIR).as_posix(),
                "full_path": str(path),
                "split": "extra",
                "label": label,
                "attack_type": attack_type,
                "source_group": source_group,
                **meta,
            }
        )

    return rows, ignored


def rows_from_dataset3() -> tuple[list[dict], Counter]:
    csv_path = DATASET3_DIR / "display_spoof.csv"
    rows: list[dict] = []
    ignored = Counter()
    if not csv_path.exists():
        ignored["missing_csv"] += 1
        return rows, ignored

    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            rel_path = Path(item["file"])
            full_path = DATASET3_DIR / rel_path
            if full_path.suffix.lower() not in VIDEO_EXTS:
                ignored["non_video"] += 1
                continue
            if not full_path.exists():
                ignored["missing_video"] += 1
                continue

            meta = video_metadata(full_path)
            rows.append(
                {
                    "source_dataset": "dataset3",
                    "relative_path": rel_path.as_posix(),
                    "full_path": str(full_path),
                    "split": "extra",
                    "label": "fake",
                    "attack_type": "display_spoof",
                    "source_group": "dataset3_display_attack",
                    **meta,
                }
            )
    return rows, ignored


def summarize(rows: list[dict], ignored: Counter) -> dict:
    by_dataset = Counter()
    by_label = Counter()
    by_split = Counter()
    by_attack = Counter()
    by_resolution = Counter()
    by_dataset_label = defaultdict(Counter)

    for row in rows:
        by_dataset[row["source_dataset"]] += 1
        by_label[row["label"]] += 1
        by_split[row["split"]] += 1
        by_attack[row["attack_type"]] += 1
        by_resolution[f'{row["width"]}x{row["height"]}'] += 1
        by_dataset_label[row["source_dataset"]][row["label"]] += 1

    return {
        "total_rows": len(rows),
        "by_dataset": dict(by_dataset),
        "by_label": dict(by_label),
        "by_split": dict(by_split),
        "by_attack_type": dict(by_attack),
        "by_resolution": dict(by_resolution),
        "by_dataset_label": {key: dict(value) for key, value in by_dataset_label.items()},
        "ignored_counts": dict(ignored),
    }


def write_manifest(rows: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_dataset",
        "relative_path",
        "full_path",
        "split",
        "label",
        "attack_type",
        "source_group",
        "opened",
        "fps",
        "frame_count",
        "width",
        "height",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(out)


def main() -> int:
    rows = rows_from_dataset1()
    dataset2_rows, ignored = rows_from_dataset2()
    dataset3_rows, ignored_dataset3 = rows_from_dataset3()
    rows.extend(dataset2_rows)
    rows.extend(dataset3_rows)
    ignored.update(ignored_dataset3)

    write_manifest(rows)
    summary = summarize(rows, ignored)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nManifest ditulis ke: {OUTPUT_CSV}")
    print(f"Ringkasan ditulis ke: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
