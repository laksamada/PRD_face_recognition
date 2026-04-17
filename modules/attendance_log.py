"""
attendance_log.py
-----------------
Pencatatan kehadiran ke CSV (satu file per hari).
Secara default, satu NIM hanya dicatat sekali per hari.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Cooldown per NIM (detik) — mencegah spam deteksi
COOLDOWN_SEC = 10
_last_recorded: dict[str, float] = {}


def record(
    nim: str,
    name: str,
    score: float,
    allow_repeat_same_day: bool = False,
) -> bool:
    """
    Catat kehadiran ke CSV harian.
    Return True jika berhasil dicatat, False jika duplikat/cooldown.
    """
    now = time.time()

    # Cek cooldown
    if nim in _last_recorded and (now - _last_recorded[nim]) < COOLDOWN_SEC:
        return False

    if not allow_repeat_same_day and nim in get_recorded_today():
        _last_recorded[nim] = now
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"attendance_{today}.csv"
    is_new_file = not log_file.exists()

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "nim", "name", "confidence"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            nim,
            name,
            f"{score:.4f}",
        ])

    _last_recorded[nim] = now
    return True


def get_recorded_today() -> set[str]:
    """Kembalikan set NIM yang sudah dicatat hari ini (dari file CSV)."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"attendance_{today}.csv"
    if not log_file.exists():
        return set()

    nims = set()
    with open(log_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nims.add(row["nim"])
    return nims
