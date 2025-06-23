@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "FINAL_SCRIPT_EXIT_CODE=0"

REM === Configuration ===
SET "SCRIPT_NAME=run_wubu.bat"
SET "AGENT_NAME=WuBu"
SET "VENV_DIR_NAME=venv"
SET "PYTHON_MAIN_SCRIPT=main.py"
SET "PROJECT_ROOT=%~dp0"
SET "VENV_ACTIVATION_SCRIPT=%PROJECT_ROOT%%VENV_DIR_NAME%\Scripts\activate.bat"
SET "AGENT_SCRIPT_PATH=%PROJECT_ROOT%%PYTHON_MAIN_SCRIPT%"

ECHO [!SCRIPT_NAME!] Starting !AGENT_NAME!...

REM --- 1. Activate Virtual Environment ---
ECHO.
ECHO [!SCRIPT_NAME!] Activating Python virtual environment...
ECHO [!SCRIPT_NAME!] Venv path: "!VENV_ACTIVATION_SCRIPT!"
IF NOT EXIST "!VENV_ACTIVATION_SCRIPT!" (
    ECHO [!SCRIPT_NAME!] FATAL ERROR: Venv script not found: "!VENV_ACTIVATION_SCRIPT!"
    SET "FINAL_SCRIPT_EXIT_CODE=1"
    GOTO :ScriptEnd
)

CALL "!VENV_ACTIVATION_SCRIPT!"
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] FATAL ERROR: CALL to venv script failed. Code: !ERRORLEVEL!
    SET "FINAL_SCRIPT_EXIT_CODE=1"
    GOTO :ScriptEnd
)
ECHO [!SCRIPT_NAME!] Virtual environment seems activated.

REM --- 2. Prepare and Run Agent ---
ECHO.
ECHO [!SCRIPT_NAME!] Preparing to start !AGENT_NAME! script...
ECHO [!SCRIPT_NAME!]   Agent Script: "!AGENT_SCRIPT_PATH!"
ECHO [!SCRIPT_NAME!]   Arguments: %*

IF NOT EXIST "!AGENT_SCRIPT_PATH!" (
    ECHO [!SCRIPT_NAME!] FATAL ERROR: Agent script "!AGENT_SCRIPT_PATH!" not found.
    SET "FINAL_SCRIPT_EXIT_CODE=1"
    GOTO :ScriptEnd
)

ECHO [!SCRIPT_NAME!] Changing to project root: "!PROJECT_ROOT!"
PUSHD "!PROJECT_ROOT!"
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] FATAL ERROR: Failed PUSHD to "!PROJECT_ROOT!". Code: !ERRORLEVEL!
    SET "FINAL_SCRIPT_EXIT_CODE=1"
    GOTO :ScriptEnd
)

ECHO [!SCRIPT_NAME!] Current directory: "%cd%"
ECHO [!SCRIPT_NAME!] Executing: python "!PYTHON_MAIN_SCRIPT!" %*
ECHO.

python "!PYTHON_MAIN_SCRIPT!" %*
SET "PYTHON_ERRORLEVEL=!ERRORLEVEL!"

ECHO.
ECHO [!SCRIPT_NAME!] Python script finished. Errorlevel from Python: !PYTHON_ERRORLEVEL!

POPD
ECHO [!SCRIPT_NAME!] Restored original directory. Current directory: "%cd%"
ECHO [DEBUG] PYTHON_ERRORLEVEL before IF is: '!PYTHON_ERRORLEVEL!'

REM Using "quotes" around the values in the IF for maximum safety
IF "!PYTHON_ERRORLEVEL!" NEQ "0" (
    ECHO [!SCRIPT_NAME!] WARNING: Python script exited with error code !PYTHON_ERRORLEVEL!.
    SET "FINAL_SCRIPT_EXIT_CODE=!PYTHON_ERRORLEVEL!"
) ELSE (
    ECHO [!SCRIPT_NAME!] Python script execution successful.
    SET "FINAL_SCRIPT_EXIT_CODE=0"
)
ECHO [DEBUG] FINAL_SCRIPT_EXIT_CODE after IF is: '!FINAL_SCRIPT_EXIT_CODE!'

REM --- 3. Finalization ---
:ScriptEnd
ECHO.
ECHO [!SCRIPT_NAME!] Script finalization.
IF "!FINAL_SCRIPT_EXIT_CODE!" EQU "0" (
    ECHO [!SCRIPT_NAME!] !AGENT_NAME! execution finished successfully.
) ELSE (
    ECHO [!SCRIPT_NAME!] !AGENT_NAME! execution finished with ERRORLEVEL: !FINAL_SCRIPT_EXIT_CODE!.
    ECHO [!SCRIPT_NAME!] Review output for details.
)

ECHO [DEBUG ScriptEnd] Exiting with code: !FINAL_SCRIPT_EXIT_CODE!
ENDLOCAL
EXIT /B %FINAL_SCRIPT_EXIT_CODE%