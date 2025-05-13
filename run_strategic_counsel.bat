@echo off
REM Batch file to run the Strategic Counsel Streamlit application

REM Get the directory where this batch file is located
SET "BATCH_DIR=%~dp0"

REM Navigate to the application directory (where app.py is)
REM This assumes the .bat file is in the same directory as app.py (e.g., SC Gen 2)
cd /d "%BATCH_DIR%"

echo Starting Strategic Counsel application...
echo Make sure you have all dependencies installed (e.g., streamlit, openai, pandas, etc.)
echo If you use a virtual environment, activate it before running this script,
echo or add the activation command below.

REM --- Optional: Activate your Python virtual environment ---
REM If you have a virtual environment (e.g., in a folder named 'venv' or '.venv'),
REM uncomment and adjust the line below.
REM Example for a venv folder named 'venv':
REM IF EXIST "venv\Scripts\activate.bat" (
REM     echo Activating virtual environment...
REM     call "venv\Scripts\activate.bat"
REM ) ELSE (
REM     echo Virtual environment 'venv' not found. Proceeding with system Python.
REM )

REM Run the Streamlit application
REM Ensure 'python' is in your system's PATH or use the full path to python.exe
python -m streamlit run app.py

echo Application stopped.
pause
