@echo off
echo Setting up virtual environment for Gender Classification ML project...
echo.

echo Creating virtual environment...
python -m venv .venv

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To activate the virtual environment in future sessions, run:
echo .\.venv\Scripts\activate.bat
echo.
echo To run the project:
echo python app.py
echo.
pause
