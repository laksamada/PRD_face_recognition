# PRD Face Recognition

Desktop app presensi mahasiswa berbasis pengenalan wajah dengan:

- registrasi wajah multi-pose
- presensi real-time dari webcam
- database embedding lokal

## Struktur

- `app.py`: aplikasi utama presensi + pendaftaran
- `modules/`: database, matcher, engine wajah, log absensi
- `data/`: scaffold data lokal saat runtime

## Setup Cepat (Windows)

Untuk aplikasi saja:

```powershell
.\setup_windows.bat
.\run.bat
```

Jika ingin install manual:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-app.txt
python -c "from modules.face_engine import get_engine; get_engine()"
```

## Requirements

- `requirements-app.txt`: dependensi runtime aplikasi

## Catatan

- Dataset besar dan artefak model tidak disertakan di repo ini.
- Folder `data/` disiapkan sebagai tempat database dan log lokal saat aplikasi berjalan.
- Folder `.venv` tidak ikut dipush karena ukurannya sangat besar dan tidak portabel lintas mesin.
- `setup_windows.bat` juga akan memicu download model InsightFace yang dibutuhkan saat first run.
