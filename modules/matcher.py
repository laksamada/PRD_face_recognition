"""
matcher.py
----------
Pencocokan wajah real-time menggunakan Cosine Similarity (vectorized).
Tidak menggunakan for-loop — pakai matrix dot-product NumPy.
"""

import numpy as np
from .database import load_database

DEFAULT_THRESHOLD = 0.55


class FaceMatcher:
    """
    Muat database sekali, lakukan pencocokan cepat di setiap frame.
    Refresh database dengan memanggil reload() jika ada pendaftaran baru.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._db_vectors: np.ndarray | None = None
        self._students: list[dict] = []
        self.reload()

    def reload(self) -> None:
        """Muat ulang database dari disk (panggil setelah ada enrollment baru)."""
        self._db_vectors, self._students = load_database()

    @property
    def is_empty(self) -> bool:
        return self._db_vectors is None or len(self._students) == 0

    def find(self, face_vector: np.ndarray) -> tuple[dict | None, float]:
        """
        Cari wajah paling mirip di database.

        Args:
            face_vector: numpy array (512,) — sudah L2-normalized.

        Returns:
            (student_dict, score) jika cocok (score >= threshold),
            (None, best_score) jika tidak ada yang cocok.
        """
        if self.is_empty:
            return None, 0.0

        valid_count = min(int(self._db_vectors.shape[0]), len(self._students))
        if valid_count <= 0:
            return None, 0.0

        # Vectorized cosine similarity: (N, 512) @ (512,) → (N,)
        # Karena semua vektor sudah L2-normalized, dot product = cosine similarity.
        similarities: np.ndarray = np.dot(self._db_vectors[:valid_count], face_vector)

        best_idx: int = int(np.argmax(similarities))
        best_score: float = float(similarities[best_idx])

        if best_score >= self.threshold:
            return self._students[best_idx], best_score
        return None, best_score
