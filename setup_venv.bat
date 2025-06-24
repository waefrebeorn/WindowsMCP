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

REM === Step 2: (Zonos Docker setup has been removed) ===
ECHO [!SCRIPT_NAME!] Zonos Docker setup step has been removed. Local Zonos is now configured differently.
ECHO [!SCRIPT_NAME!] For Windows users wanting Zonos Local TTS, please use 'install_uv_qinglong.ps1'.
ECHO.

REM === Main Completion ===
ECHO [!SCRIPT_NAME!] Basic Python environment setup process has been called.
GOTO :HandleMasterExit

REM === Final Exit Point for Master Script ===
:HandleMasterExit
ECHO.
IF "!MASTER_SCRIPT_EXIT_CODE!" NEQ "0" (
    ECHO [!SCRIPT_NAME!] Basic Python environment setup finished with errors from 'setup_python_env.bat'. Error Code: !MASTER_SCRIPT_EXIT_CODE!
) ELSE (
    ECHO [!SCRIPT_NAME!] Basic Python environment setup finished successfully (via 'setup_python_env.bat').
)
ECHO.
ECHO [!SCRIPT_NAME!] Review the output above for details on the Python environment setup.
ECHO [!SCRIPT_NAME!] For Zonos Local TTS and advanced features on Windows, it is recommended to use 'install_uv_qinglong.ps1'.
ENDLOCAL & (
    PAUSE
    EXIT /B %MASTER_SCRIPT_EXIT_CODE%
)
