@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "MASTER_SCRIPT_EXIT_CODE=0"
SET "SCRIPT_NAME=setup_venv.bat"

ECHO [!SCRIPT_NAME!] Starting main setup process...
ECHO.

REM === Step 1: Setup Python Environment ===
ECHO [!SCRIPT_NAME!] Calling Python environment setup script (setup_python_env.bat)...
CALL "%~dp0setup_python_env.bat"
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Python environment setup (setup_python_env.bat) failed. See messages above.
    SET "MASTER_SCRIPT_EXIT_CODE=!ERRORLEVEL!"
    GOTO :HandleMasterExit
)
ECHO [!SCRIPT_NAME!] Python environment setup script completed.
ECHO.

REM === Step 2: Setup Zonos Docker Environment ===
ECHO [!SCRIPT_NAME!] Calling Zonos Docker setup script (setup_docker_zonos.bat)...
CALL "%~dp0setup_docker_zonos.bat"
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Zonos Docker setup (setup_docker_zonos.bat) failed. See messages above.
    SET "MASTER_SCRIPT_EXIT_CODE=!ERRORLEVEL!"
    GOTO :HandleMasterExit
)
ECHO [!SCRIPT_NAME!] Zonos Docker setup script completed.
ECHO.

REM === Main Completion ===
ECHO [!SCRIPT_NAME!] All setup processes have been called.
GOTO :HandleMasterExit

REM === Final Exit Point for Master Script ===
:HandleMasterExit
ECHO.
IF "!MASTER_SCRIPT_EXIT_CODE!" NEQ "0" (
    ECHO [!SCRIPT_NAME!] Master setup script finished with errors. Error Code: !MASTER_SCRIPT_EXIT_CODE!
) ELSE (
    ECHO [!SCRIPT_NAME!] Master setup script finished successfully.
)
ECHO.
ECHO [!SCRIPT_NAME!] Review the output above for details on each step.
ENDLOCAL & (
    PAUSE
    EXIT /B %MASTER_SCRIPT_EXIT_CODE%
)
