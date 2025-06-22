@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_EXIT_CODE=0"

REM === Configuration ===
SET "VENV_DIR=venv"
SET "REQUIREMENTS_FILE=requirements.txt"
SET "SCRIPT_NAME=setup_venv.bat"

ECHO [!SCRIPT_NAME!] Starting Python Virtual Environment Setup...

REM === Validation ===
CALL :CheckPython
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Python check failed. Script will terminate.
    SET "SCRIPT_EXIT_CODE=!ERRORLEVEL!"
    GOTO :HandleExit
)

REM === Main Logic ===
CALL :CreateVenv
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Virtual environment creation failed. Script will terminate.
    SET "SCRIPT_EXIT_CODE=!ERRORLEVEL!"
    GOTO :HandleExit
)

CALL :InstallDependencies
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Failed to install dependencies. Script will terminate.
    SET "SCRIPT_EXIT_CODE=!ERRORLEVEL!"
    GOTO :HandleExit
)

REM === Completion (if all successful) ===
ECHO.
ECHO =====================================================================
ECHO [!SCRIPT_NAME!] Setup complete!
ECHO Virtual environment "!VENV_DIR!" is ready and dependencies are installed.
ECHO To activate the virtual environment in your current shell, run:
ECHO   %~dp0!VENV_DIR!\Scripts\activate.bat
ECHO =====================================================================
SET "SCRIPT_EXIT_CODE=0" REM Explicitly set for success path
GOTO :HandleExit


REM === Subroutines ===
:CheckPython
    ECHO [!SCRIPT_NAME!] Checking for Python installation...
    python --version >nul 2>&1
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Python is not installed or not found in your system's PATH.
        ECHO [!SCRIPT_NAME!] Please install Python from python.org and ensure it's added to PATH.
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Python found.
EXIT /B 0

:CreateVenv
    ECHO.
    ECHO [!SCRIPT_NAME!] Checking for virtual environment directory: "!VENV_DIR!"
    IF EXIST "%~dp0!VENV_DIR!\Scripts\activate.bat" (
        ECHO [!SCRIPT_NAME!] Virtual environment "!VENV_DIR!" already exists and appears valid. Skipping creation.
        EXIT /B 0
    )

    IF EXIST "%~dp0!VENV_DIR!" (
        ECHO [!SCRIPT_NAME!] Directory "%~dp0!VENV_DIR!" exists but doesn't seem to be a valid venv.
        ECHO [!SCRIPT_NAME!] You might need to remove it manually if you want to recreate it.
        ECHO [!SCRIPT_NAME!] For now, attempting to proceed assuming it might be usable or will be fixed by Python.
    )

    ECHO [!SCRIPT_NAME!] Creating virtual environment in "%~dp0!VENV_DIR!"...
    python -m venv "%~dp0!VENV_DIR!"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to create virtual environment in "%~dp0!VENV_DIR!".
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Virtual environment created.
EXIT /B 0

:InstallDependencies
    ECHO.
    ECHO [!SCRIPT_NAME!] Activating virtual environment to install dependencies...
    IF NOT EXIST "%~dp0!VENV_DIR!\Scripts\activate.bat" (
        ECHO [!SCRIPT_NAME!] ERROR: Cannot find activate script at "%~dp0!VENV_DIR!\Scripts\activate.bat".
        ECHO [!SCRIPT_NAME!] Virtual environment setup might have failed.
        EXIT /B 1
    )
    CALL "%~dp0!VENV_DIR!\Scripts\activate.bat"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to call the activate script. Venv might be corrupted or not found.
        EXIT /B 1
    )

    ECHO [!SCRIPT_NAME!] Installing dependencies from "%~dp0!REQUIREMENTS_FILE!"...
    IF NOT EXIST "%~dp0!REQUIREMENTS_FILE!" (
        ECHO [!SCRIPT_NAME!] ERROR: "%~dp0!REQUIREMENTS_FILE!" not found. Cannot install dependencies.
        EXIT /B 1
    )
    pip install -r "%~dp0!REQUIREMENTS_FILE!"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to install dependencies from "%~dp0!REQUIREMENTS_FILE!".
        ECHO [!SCRIPT_NAME!] Check your internet connection and the contents of the file.
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Dependencies installed successfully.

    REM Deactivate the venv after installing dependencies if this script is not meant to keep it active
    ECHO [!SCRIPT_NAME!] Deactivating virtual environment (installation complete).
    CALL :DeactivateVenvQuietly
EXIT /B 0

:DeactivateVenvQuietly
    REM This is a bit of a trick as 'deactivate' is a function defined by activate.bat
    REM and not a standalone script. Calling it directly might not always work as expected
    REM or might output "The system cannot find the path specified." if not careful.
    REM For robustness, it's often better to just let the SETLOCAL/ENDLOCAL handle environment restoration.
    REM However, if an explicit deactivate is desired:
    IF DEFINED VIRTUAL_ENV (
        REM Check if deactivate function exists (it's a bit hacky to check this way)
        REM A more reliable way is usually not needed as ENDLOCAL handles cleanup.
        REM For now, we'll skip a direct "call deactivate" as it can be problematic.
        ECHO [!SCRIPT_NAME!] Venv was active. Script's ENDLOCAL will handle restoration.
    )
EXIT /B 0

REM === Final Exit Point ===
:HandleExit
ECHO.
IF "!SCRIPT_EXIT_CODE!" NEQ "0" (
    ECHO [!SCRIPT_NAME!] Script finished with errors. Error Code: !SCRIPT_EXIT_CODE!
) ELSE (
    ECHO [!SCRIPT_NAME!] Script finished successfully.
)
ENDLOCAL & (
    PAUSE
    EXIT /B %SCRIPT_EXIT_CODE%
)
