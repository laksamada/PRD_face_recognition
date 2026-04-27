@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
python -c "from modules.face_engine import get_engine; get_engine(); print('InsightFace model siap.')"

echo.
echo Environment app siap.
echo Jalankan: .\run.bat
pause
