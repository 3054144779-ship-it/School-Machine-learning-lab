@echo off
title Student Score Predict System

set "ROOT=%~dp0"

echo ============================================
echo   Student Score Predict System
echo ============================================
echo.
echo Root: %ROOT%
echo.

REM --- Check and train model ---
if not exist "%ROOT%score-predict-model\saved_models" (
    echo [1/3] Training model, please wait...
    cd /d "%ROOT%score-predict-model"
    python main.py
    if errorlevel 1 (
        echo ERROR: Model training failed!
        pause
        exit /b 1
    )
    cd /d "%ROOT%"
    echo Training done.
) else (
    echo [1/3] Model found, skip training.
)

REM --- Start Python API ---
echo [2/3] Starting Python API on port 5000...
start "Python-API" cmd /c "cd /d %ROOT%score-predict-model && python api.py && pause"

REM --- Start Java Backend ---
echo [2/3] Starting Java Backend on port 8080...
start "Java-Backend" cmd /c "cd /d %ROOT%score-predict-backend && mvnw.cmd spring-boot:run -q && pause"

REM --- Start Frontend ---
echo [3/3] Starting Frontend on port 3000...
start "Frontend" cmd /c "cd /d %ROOT%score-predict-frontend\student_score_predict && npm run dev && pause"

echo.
echo ============================================
echo   All services launching in separate windows.
echo   Python API : http://localhost:5000
echo   Backend   : http://localhost:8080
echo   Frontend  : http://localhost:3000
echo ============================================
echo.
pause
