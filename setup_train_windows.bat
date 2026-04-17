@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu130

echo.
echo Environment training siap.
echo App runtime + paket training sudah terpasang.
pause
