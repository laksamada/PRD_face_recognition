# PRD Face Recognition

Desktop app presensi mahasiswa berbasis pengenalan wajah dengan:

- registrasi wajah multi-pose
- presensi real-time dari webcam
- database embedding lokal
- pipeline eksperimen anti-spoofing berbasis video

## Struktur

- `app.py`: aplikasi utama presensi + pendaftaran
- `modules/`: database, matcher, engine wajah, log absensi
- `anti_spoofing/`: script preprocessing dan training anti-spoofing
- `data/`: scaffold data lokal saat runtime

## Setup Cepat (Windows)

Untuk aplikasi saja:

```powershell
.\setup_windows.bat
.\run.bat
```

Untuk aplikasi + environment training anti-spoofing:

```powershell
.\setup_train_windows.bat
```

Jika ingin install manual:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-app.txt
```

## Requirements

- `requirements-app.txt`: dependensi runtime aplikasi
- `requirements-train.txt`: runtime + dependensi training
- `requirements-lock.txt`: snapshot penuh dari environment lokal pengembangan

## Catatan

- Dataset besar dan artefak model tidak disertakan di repo ini.
- Folder `data/` disiapkan sebagai tempat database dan log lokal saat aplikasi berjalan.
- Folder `.venv` tidak ikut dipush karena ukurannya sangat besar dan tidak portabel lintas mesin.
