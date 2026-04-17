"""
database.py
-----------
Pengelola penyimpanan vektor wajah (face_vectors.npy) dan metadata (student_data.json).
Selalu pastikan indeks JSON sinkron dengan baris matriks .npy.
"""

import json
import numpy as np
from pathlib import Path

VECTORS_PATH = Path(__file__).parent.parent / "data" / "face_vectors.npy"
METADATA_PATH = Path(__file__).parent.parent / "data" / "student_data.json"


def _load_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"students": []}


def _save_metadata(data: dict) -> None:
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_vectors_raw() -> np.ndarray | None:
    if VECTORS_PATH.exists():
        vectors = np.load(str(VECTORS_PATH))
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors.astype(np.float32, copy=False)
    return None


def _save_vectors(vectors: np.ndarray | None) -> None:
    if vectors is None or len(vectors) == 0:
        if VECTORS_PATH.exists():
            VECTORS_PATH.unlink()
        return
    np.save(str(VECTORS_PATH), vectors.astype(np.float32, copy=False))


def _normalize_students(students: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for idx, student in enumerate(students):
        normalized.append(
            {
                **student,
                "index": idx,
            }
        )
    return normalized


def load_database() -> tuple[np.ndarray | None, list[dict]]:
    """
    Muat database wajah dan sinkronkan otomatis jika jumlah vector/metadata berbeda.

    Jika pernah terjadi crash di tengah penulisan, file .npy bisa lebih panjang
    daripada metadata JSON. Dalam kasus itu, kita pertahankan irisan yang aman.
    """
    vectors = _load_vectors_raw()
    metadata = _load_metadata()
    students = metadata.get("students", [])
    if not isinstance(students, list):
        students = []

    vector_count = 0 if vectors is None else int(vectors.shape[0])
    student_count = len(students)
    synced_count = min(vector_count, student_count)
    metadata_changed = False

    if vector_count != student_count:
        print(
            "[Database] warning: jumlah vector dan metadata tidak sinkron "
            f"({vector_count} vs {student_count}). Menyelaraskan ke {synced_count}."
        )
        if synced_count == 0:
            vectors = None
            students = []
        else:
            vectors = vectors[:synced_count].copy() if vectors is not None else None
            students = students[:synced_count]
        metadata_changed = True

    normalized_students = _normalize_students(students)
    if normalized_students != students:
        metadata_changed = True

    if metadata_changed:
        _save_vectors(vectors)
        _save_metadata({"students": normalized_students})

    return vectors, normalized_students


def load_vectors() -> np.ndarray | None:
    """Muat matriks vektor (N, 512). Return None jika belum ada data."""
    vectors, _ = load_database()
    return vectors


def load_students() -> list[dict]:
    """Kembalikan list metadata mahasiswa."""
    _, students = load_database()
    return students


def add_student(nim: str, name: str, embedding: np.ndarray) -> int:
    """
    Tambahkan mahasiswa baru ke database.

    Urutan operasi:
    1. Append vektor ke .npy terlebih dahulu.
    2. Baru append metadata ke .json.

    Return: index baru mahasiswa.
    """
    embedding = embedding.astype(np.float32)
    vectors, students = load_database()

    # Step 1: simpan vektor
    if vectors is None:
        new_matrix = embedding.reshape(1, -1)
    else:
        new_matrix = np.vstack([vectors, embedding.reshape(1, -1)])
    _save_vectors(new_matrix)

    # Step 2: simpan metadata (hanya jika step 1 berhasil)
    new_index = len(students)
    students.append({"index": new_index, "nim": nim, "name": name})
    _save_metadata({"students": _normalize_students(students)})

    return new_index


def is_nim_registered(nim: str) -> bool:
    """Cek apakah NIM sudah terdaftar."""
    return any(s["nim"] == nim for s in load_students())


def get_student_count() -> int:
    return len(load_students())
