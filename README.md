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

## Menjalankan

Siapkan virtual environment Python yang berisi dependensi proyek, lalu jalankan:

```powershell
python app.py
```

Atau di Windows:

```powershell
.\run.bat
```

## Catatan

- Dataset besar dan artefak model tidak disertakan di repo ini.
- Folder `data/` disiapkan sebagai tempat database dan log lokal saat aplikasi berjalan.
