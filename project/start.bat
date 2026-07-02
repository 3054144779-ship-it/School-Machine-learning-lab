@echo off
setlocal enabledelayedexpansion
title Student Score Predict System

set "ROOT=%~dp0"

echo ============================================
echo   Student Score Predict System
echo ============================================
echo.
echo Root: %ROOT%
echo.

REM =============================================
REM  CONDA ENVIRONMENT (edit this if using conda)
REM  Set CONDA_ENV to your conda environment name
REM  Leave empty to use whatever python is on PATH
REM =============================================
set "CONDA_ENV=new_env"
REM Example: set "CONDA_ENV=ml_env"

REM =============================================
REM  AUTO-DETECT PYTHON (conda env > .venv > PATH)
REM =============================================

REM --- Find conda ---
set "CONDA="
where conda >nul 2>&1
if !errorlevel!==0 (
    for /f "delims=" %%x in ('where conda 2^>nul') do set "CONDA=%%x"
)
if defined CONDA echo Conda: !CONDA!

if not "%CONDA_ENV%"=="" (
    if defined CONDA (
        echo Activating conda environment: %CONDA_ENV%
        call "%CONDA%" activate %CONDA_ENV% 2>nul
        if !errorlevel!==0 (
            echo Conda env '%CONDA_ENV%' activated.
            REM Use 'where python' to get the active env's python.exe
            for /f "delims=" %%x in ('where python 2^>nul') do (
                if not defined PYTHON (
                    for %%n in ("%%x") do set "PYTHON=%%~fn"
                )
            )
            if defined PYTHON goto python_found
        )
        echo WARNING: Failed to activate '%CONDA_ENV%', falling back...
    ) else (
        echo WARNING: conda not found, cannot activate '%CONDA_ENV%'.
    )
)

REM --- Find venv in project ---
if exist "%ROOT%score-predict-model\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%score-predict-model\.venv\Scripts\python.exe"
    set "PIP=%ROOT%score-predict-model\.venv\Scripts\pip.exe"
    echo Using project .venv: %PYTHON%
    goto python_found
)

REM --- Find Python from PATH ---
set "PYTHON="
for %%p in (python python3) do (
    where %%p >nul 2>&1
    if !errorlevel!==0 (
        for /f "delims=" %%x in ('where %%p 2^>nul') do (
            if not defined PYTHON (
                REM Resolve to full path
                for %%n in ("%%x") do set "PYTHON=%%~fn"
            )
        )
    )
)
if not defined PYTHON (
    REM Fallback: scan common Python install dirs
    for %%d in (
        "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312"
        "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313"
        "C:\Python312" "C:\Python313"
        "D:\Anaconda" "C:\Anaconda"
        "D:\ProgramData\anaconda3" "C:\ProgramData\anaconda3"
    ) do (
        if exist "%%~d\python.exe" if not defined PYTHON (
            for %%n in ("%%~d\python.exe") do set "PYTHON=%%~fn"
        )
    )
)
if not defined PYTHON (
    echo ERROR: Python not found. Please install Python 3.10+ and add it to PATH.
    echo Or edit this script and set PYTHON= to your python.exe path.
    pause & exit /b 1
)

:python_found

REM --- Find pip (same dir as python) ---
for %%x in ("%PYTHON%") do set "PYDIR=%%~dpx"
if exist "%PYDIR%pip.exe"         set "PIP=%PYDIR%pip.exe"
if exist "%PYDIR%pip3.exe"        set "PIP=%PYDIR%pip3.exe"
if exist "%PYDIR%Scripts\pip.exe" set "PIP=%PYDIR%Scripts\pip.exe"
if not defined PIP set "PIP=%PYTHON% -m pip"

echo Python: %PYTHON%
echo Pip:    %PIP%

REM --- Find MySQL ---
set "MYSQLD="
set "MYSQLADMIN="
set "MYSQLCLI="

REM Try find from PATH first
where mysql >nul 2>&1
if !errorlevel!==0 (
    for /f "delims=" %%x in ('where mysql 2^>nul') do set "MYSQLCLI=%%x"
    for %%x in ("%MYSQLCLI%") do set "MYSQLDIR=%%~dpx"
    if exist "!MYSQLDIR!mysqld.exe"     set "MYSQLD=!MYSQLDIR!mysqld.exe"
    if exist "!MYSQLDIR!mysqladmin.exe" set "MYSQLADMIN=!MYSQLDIR!mysqladmin.exe"
)

REM Fallback: scan common MySQL install dirs
if not defined MYSQLD (
    for %%v in (8.4 8.0 8.3 9.0) do (
        for %%d in ("C:\Program Files\MySQL\MySQL Server %%v\bin") do (
            if exist "%%~d\mysqld.exe" if not defined MYSQLD (
                set "MYSQLD=%%~d\mysqld.exe"
                set "MYSQLADMIN=%%~d\mysqladmin.exe"
                set "MYSQLCLI=%%~d\mysql.exe"
            )
        )
    )
)

if not defined MYSQLD (
    echo WARNING: MySQL not auto-detected. Will skip MySQL startup.
    echo If you have MySQL, edit this script and set MYSQLD/MYSQLADMIN/MYSQLCLI.
    set "HAS_MYSQL=0"
) else (
    echo MySQL: %MYSQLD%
    set "HAS_MYSQL=1"
)

echo.

REM =============================================
REM  CLEAN UP EXISTING SERVICES ON PORTS
REM =============================================
echo Cleaning up existing services...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5000 " ^| findstr "LISTENING"') do taskkill //F //PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8080 " ^| findstr "LISTENING"') do taskkill //F //PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":3000 " ^| findstr "LISTENING"') do taskkill //F //PID %%a >nul 2>&1
echo Done.
echo.

REM =============================================
REM  STEP 0: INSTALL PYTHON DEPENDENCIES
REM =============================================
echo [Setup] Checking Python dependencies...
"%PYTHON%" -c "import fastapi, uvicorn, joblib, numpy, pandas, sklearn, openpyxl, pydantic, pymysql" >nul 2>&1
if !errorlevel!==0 (
    echo Dependencies OK.
) else (
    echo Installing missing dependencies...
    "%PIP%" install -q fastapi uvicorn joblib numpy pandas scikit-learn openpyxl pydantic pymysql >nul 2>&1
    if errorlevel 1 (
        "%PIP%" install -q --user fastapi uvicorn joblib numpy pandas scikit-learn openpyxl pydantic pymysql >nul 2>&1
    )
    echo Done.
)
echo.

REM =============================================
REM  STEP 0.5: START MYSQL (if available)
REM =============================================
if "%HAS_MYSQL%"=="0" goto skip_mysql

echo [0/5] Starting MySQL...
"%MYSQLADMIN%" ping -h 127.0.0.1 -u root -proot123456 >nul 2>&1
if !errorlevel!==0 (
    echo MySQL already running.
    goto skip_mysql
)

echo Starting MySQL server...
start "MySQL" /MIN "%MYSQLD%" --console

set /a mysql_wait=0
:mysql_wait_loop
ping -n 2 127.0.0.1 >nul
set /a mysql_wait+=1
"%MYSQLADMIN%" ping -h 127.0.0.1 -u root -proot123456 >nul 2>&1
if !errorlevel!==0 (
    echo MySQL started.
    goto skip_mysql
)
if %mysql_wait% LSS 30 goto mysql_wait_loop
echo WARNING: MySQL may not have started, continuing anyway...

:skip_mysql
echo.

REM =============================================
REM  STEP 1: TRAIN MODEL (if needed)
REM =============================================
if not exist "%ROOT%score-predict-model\saved_models" (
    echo [1/5] Training model, please wait...
    cd /d "%ROOT%score-predict-model"
    "%PYTHON%" main.py
    if errorlevel 1 (
        echo ERROR: Model training failed!
        pause
        exit /b 1
    )
    cd /d "%ROOT%"
    echo [1/5] Training done.
) else (
    echo [1/5] Model found, skip training.
)
echo.

REM =============================================
REM  STEP 2: IMPORT DATA TO MYSQL (if needed)
REM =============================================
echo [2/5] Checking database...
if "%HAS_MYSQL%"=="0" (
    echo [2/5] MySQL not available, skip database check.
    goto skip_db
)

"%MYSQLCLI%" -h 127.0.0.1 -u root -proot123456 -e "CREATE DATABASE IF NOT EXISTS student_predict DEFAULT CHARACTER SET utf8mb4;" >nul 2>&1
"%MYSQLCLI%" -h 127.0.0.1 -u root -proot123456 -e "SELECT COUNT(*) FROM student_predict.t_student_history;" >nul 2>&1
if errorlevel 1 (
    echo [2/5] Database not ready, importing data...
    cd /d "%ROOT%score-predict-model"
    "%PYTHON%" import_to_db.py
    cd /d "%ROOT%"
) else (
    echo [2/5] Database OK.
)

:skip_db
echo.

REM =============================================
REM  STEP 3: START PYTHON API (:5000)
REM =============================================
echo [3/5] Starting Python API on port 5000...
start "Python-API" cmd /c "title Python-API & cd /d "%ROOT%score-predict-model" & echo Starting FastAPI... & "%PYTHON%" api.py & pause"

echo Waiting for Python API (max 30s)...
set /a py_wait=0
:py_wait_loop
ping -n 2 127.0.0.1 >nul
set /a py_wait+=1
netstat -ano 2>nul | findstr /c:":5000 " | findstr /c:"LISTENING" >nul 2>&1
if !errorlevel!==0 (
    echo [3/5] Python API is ready.
    goto start_backend
)
if %py_wait% LSS 30 goto py_wait_loop
echo ERROR: Python API failed to start!
pause
exit /b 1

:start_backend
echo.

REM =============================================
REM  STEP 4: START JAVA BACKEND (:8080)
REM =============================================
echo [4/5] Starting Java Backend on port 8080...
echo Note: First-time build may take 2-3 minutes.

REM Use mvn if on PATH, otherwise mvnw.cmd
where mvn >nul 2>&1
if !errorlevel!==0 (
    set "MVN_CMD=mvn"
    echo Using system Maven.
) else (
    if exist "%ROOT%score-predict-backend\mvnw.cmd" (
        set "MVN_CMD=mvnw.cmd"
        echo Using Maven Wrapper.
    ) else (
        echo ERROR: Neither 'mvn' nor 'mvnw.cmd' found!
        pause & exit /b 1
    )
)

start "Java-Backend" cmd /c "title Java-Backend & cd /d "%ROOT%score-predict-backend" & echo Building and starting Spring Boot... & call !MVN_CMD! spring-boot:run & pause"

echo Waiting for Java Backend (max 180s)...
set /a jv_wait=0
:jv_wait_loop
ping -n 2 127.0.0.1 >nul
set /a jv_wait+=1
netstat -ano 2>nul | findstr /c:":8080 " | findstr /c:"LISTENING" >nul 2>&1
if !errorlevel!==0 (
    echo [4/5] Java Backend is ready.
    goto start_frontend
)
if %jv_wait% LSS 180 goto jv_wait_loop
echo ERROR: Java Backend failed to start! Check the Java-Backend window.
pause
exit /b 1

:start_frontend
echo.

REM =============================================
REM  STEP 5: START FRONTEND (:3000)
REM =============================================
echo [5/5] Starting Frontend on port 3000...

REM Check npm / node
where npm >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm not found. Please install Node.js ^>^=22.18.
    pause & exit /b 1
)

REM Install npm dependencies if missing
if not exist "%ROOT%score-predict-frontend\student_score_predict\node_modules" (
    echo Installing frontend dependencies...
    cd /d "%ROOT%score-predict-frontend\student_score_predict"
    call npm install
    cd /d "%ROOT%"
)

start "Frontend" cmd /c "title Frontend & cd /d "%ROOT%score-predict-frontend\student_score_predict" & echo Starting Vite dev server... & npm run dev & pause"

echo Waiting for Frontend (max 30s)...
set /a fe_wait=0
:fe_wait_loop
ping -n 2 127.0.0.1 >nul
set /a fe_wait+=1
netstat -ano 2>nul | findstr /c:":3000 " | findstr /c:"LISTENING" >nul 2>&1
if !errorlevel!==0 (
    echo [5/5] Frontend is ready.
    goto finish
)
if %fe_wait% LSS 30 goto fe_wait_loop
echo WARNING: Frontend may still be starting, check the Frontend window.

:finish
echo.
echo ============================================
echo   All services should be running!
echo.
echo   Python API : http://localhost:5000
echo   Backend   : http://localhost:8080
echo   Frontend  : http://localhost:3000
echo.
echo   Open http://localhost:3000 in browser.
echo ============================================
echo.
pause
exit /b 0
