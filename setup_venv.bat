@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "MASTER_SCRIPT_EXIT_CODE=0"
SET "SCRIPT_NAME=setup_venv.bat"

ECHO [!SCRIPT_NAME!] Starting WuBu Environment Setup for Windows...
ECHO [!SCRIPT_NAME!] This script will attempt to run 'install_uv_qinglong.ps1'
ECHO [!SCRIPT_NAME!] which handles the complete setup including Zonos TTS prerequisites.
ECHO.

REM Check for Administrator Privileges
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Administrator privileges are required.
    ECHO [!SCRIPT_NAME!] Please re-run this script as an Administrator.
    ECHO [!SCRIPT_NAME!] (Right-click -> Run as administrator)
    SET "MASTER_SCRIPT_EXIT_CODE=1"
    GOTO :HandleMasterExit
) ELSE (
    ECHO [!SCRIPT_NAME!] Administrator privileges detected. Proceeding...
)
ECHO.

REM === Execute PowerShell Setup Script ===
SET "POWERSHELL_SCRIPT_PATH=%~dp0install_uv_qinglong.ps1"

IF NOT EXIST "!POWERSHELL_SCRIPT_PATH!" (
    ECHO [!SCRIPT_NAME!] ERROR: PowerShell setup script not found at:
    ECHO [!SCRIPT_NAME!]   !POWERSHELL_SCRIPT_PATH!
    ECHO [!SCRIPT_NAME!] Please ensure the script exists in the same directory.
    SET "MASTER_SCRIPT_EXIT_CODE=1"
    GOTO :HandleMasterExit
)

ECHO [!SCRIPT_NAME!] Executing PowerShell setup script: !POWERSHELL_SCRIPT_PATH!
ECHO [!SCRIPT_NAME!] This may take a while. Please be patient and follow any on-screen prompts from the PowerShell script.
ECHO.
REM ExecutionPolicy Bypass is used here to ensure the script runs,
REM though install_uv_qinglong.ps1 itself also tries to install uv using -ExecutionPolicy ByPass for its sub-call.
powershell.exe -ExecutionPolicy Bypass -File "!POWERSHELL_SCRIPT_PATH!"
SET "POWERSHELL_EXIT_CODE=!ERRORLEVEL!"

IF "!POWERSHELL_EXIT_CODE!" NEQ "0" (
    ECHO [!SCRIPT_NAME!] ERROR: PowerShell setup script ('install_uv_qinglong.ps1') failed.
    ECHO [!SCRIPT_NAME!] Exit Code: !POWERSHELL_EXIT_CODE!
    ECHO [!SCRIPT_NAME!] Please review the PowerShell script output above for details.
    SET "MASTER_SCRIPT_EXIT_CODE=!POWERSHELL_EXIT_CODE!"
) ELSE (
    ECHO [!SCRIPT_NAME!] PowerShell setup script completed successfully.
)
ECHO.

REM === Main Completion ===
GOTO :HandleMasterExit

REM === Final Exit Point for Master Script ===
:HandleMasterExit
ECHO.
IF "!MASTER_SCRIPT_EXIT_CODE!" NEQ "0" (
    ECHO [!SCRIPT_NAME!] WuBu Environment Setup finished with errors. Error Code: !MASTER_SCRIPT_EXIT_CODE!
) ELSE (
    ECHO [!SCRIPT_NAME!] WuBu Environment Setup finished.
)
ECHO.
ECHO [!SCRIPT_NAME!] Review the output above for details on each step.
ENDLOCAL & (
    PAUSE
    EXIT /B %MASTER_SCRIPT_EXIT_CODE%
)
