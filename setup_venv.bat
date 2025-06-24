@echo OFF
SETLOCAL
SET "SCRIPT_NAME=setup_venv.bat"

ECHO [!SCRIPT_NAME!] WuBu Environment Setup Information
ECHO ==================================================
ECHO.
ECHO This script previously attempted to automate the full setup.
ECHO However, to ensure correct permissions and simplify the process,
ECHO the main environment setup is now handled by 'install_uv_qinglong.ps1'.
ECHO.
ECHO PLEASE NOTE:
ECHO -----------
ECHO You MUST run the PowerShell script 'install_uv_qinglong.ps1' MANUALLY
ECHO with Administrator privileges to set up the Python environment,
ECHO install dependencies (like eSpeak NG for Zonos TTS), and configure other
ECHO prerequisites.
ECHO.
ECHO To do this:
ECHO 1. Locate 'install_uv_qinglong.ps1' in the project directory.
ECHO 2. Right-click on 'install_uv_qinglong.ps1'.
ECHO 3. Select "Run as administrator".
ECHO 4. Follow any prompts from that script.
ECHO.
ECHO This 'setup_venv.bat' script no longer performs these actions directly.
ECHO Its purpose is now to guide you to use 'install_uv_qinglong.ps1'.
ECHO.
ECHO If you have already successfully run 'install_uv_qinglong.ps1' as administrator,
ECHO your environment should be ready.
ECHO.

PAUSE
EXIT /B 0
ENDLOCAL
