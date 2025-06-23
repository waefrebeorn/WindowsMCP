@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "TOPLEVEL_SCRIPT_EXIT_CODE=0"

REM === Configuration ===
SET "SCRIPT_NAME=run_ollama_wubu.bat"
SET "AGENT_NAME=WuBu"
SET "VENV_DIR_NAME=venv"
SET "PYTHON_MAIN_SCRIPT=main.py"
SET "PROJECT_ROOT=%~dp0"
SET "VENV_ACTIVATION_SCRIPT=%PROJECT_ROOT%%VENV_DIR_NAME%\Scripts\activate.bat"
SET "AGENT_SCRIPT_PATH=%PROJECT_ROOT%%PYTHON_MAIN_SCRIPT%"

ECHO [!SCRIPT_NAME!] Starting !AGENT_NAME! with Ollama provider...

REM === Main Logic ===
CALL :MainLogic %*
SET "TOPLEVEL_SCRIPT_EXIT_CODE=!ERRORLEVEL!"

REM === Final Exit Point ===
ECHO.
ECHO [!SCRIPT_NAME!] Script execution finished.
ECHO [!SCRIPT_NAME!] Final exit code for this script will be: !TOPLEVEL_SCRIPT_EXIT_CODE!
ECHO [!SCRIPT_NAME!] If this is non-zero, an error occurred. Review Python script output.
PAUSE "[!SCRIPT_NAME!] Press any key to exit..."
ENDLOCAL
EXIT /B %TOPLEVEL_SCRIPT_EXIT_CODE%

REM ============================================================================
REM === SUBROUTINES ============================================================
REM ============================================================================

:MainLogic
    SETLOCAL ENABLEDELAYEDEXPANSION
    SET "SUB_EXIT_CODE=0" 

    CALL :ActivateVenv
    IF "!ERRORLEVEL!" NEQ "0" (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to activate virtual environment.
        SET "SUB_EXIT_CODE=1"
        GOTO :EndOfMainLogic_DirectExit
    )

    ECHO [!SCRIPT_NAME!] NOTE: Ollama installation and service checks have been removed from this script.
    ECHO [!SCRIPT_NAME!] Please ensure Ollama is installed, running, and accessible for the Python script.
    ECHO [!SCRIPT_NAME!] If Ollama is not available, the Python script will likely fail.

    CALL :RunAgent %*
    SET "SUB_EXIT_CODE=!ERRORLEVEL!"
    ECHO [DEBUG MainLogic] Subroutine :RunAgent exited with code: !SUB_EXIT_CODE!
    REM Removed IF/ELSE block that was printing success/failure messages here

:EndOfMainLogic_DirectExit
    ECHO [DEBUG MainLogic] Exiting :MainLogic with code: !SUB_EXIT_CODE!
    ENDLOCAL & (
        EXIT /B %SUB_EXIT_CODE%
    )

REM ----------------------------------------------------------------------------
:ActivateVenv
    SETLOCAL
    ECHO.
    ECHO [!SCRIPT_NAME!] Activating Python virtual environment...
    ECHO [!SCRIPT_NAME!] Venv path: "!VENV_ACTIVATION_SCRIPT!"
    IF NOT EXIST "!VENV_ACTIVATION_SCRIPT!" (
        ECHO [!SCRIPT_NAME!] FATAL ERROR: Venv activation script not found.
        EXIT /B 1
    )
    CALL "!VENV_ACTIVATION_SCRIPT!"
    IF "!ERRORLEVEL!" NEQ "0" (
        ECHO [!SCRIPT_NAME!] FATAL ERROR: CALL to venv script failed.
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Virtual environment seems activated.
    ENDLOCAL
EXIT /B 0

REM ----------------------------------------------------------------------------
:RunAgent
    SETLOCAL ENABLEDELAYEDEXPANSION
    SET "PYTHON_EXEC_ERRORLEVEL=0" 
    ECHO.
    ECHO [!SCRIPT_NAME!] Preparing to start !AGENT_NAME! with Ollama provider.
    ECHO [!SCRIPT_NAME!]   Agent Script: "!AGENT_SCRIPT_PATH!"
    ECHO [!SCRIPT_NAME!]   Arguments: %*

    IF NOT EXIST "!AGENT_SCRIPT_PATH!" (
        ECHO [!SCRIPT_NAME!] FATAL ERROR: Agent script "!AGENT_SCRIPT_PATH!" not found.
        SET "PYTHON_EXEC_ERRORLEVEL=1"
        GOTO EndOfRunAgentNoPop
    )

    ECHO [!SCRIPT_NAME!] Changing to project root: "!PROJECT_ROOT!"
    PUSHD "!PROJECT_ROOT!"
    IF "!ERRORLEVEL!" NEQ "0" (
        ECHO [!SCRIPT_NAME!] FATAL ERROR: Failed PUSHD to "!PROJECT_ROOT!".
        SET "PYTHON_EXEC_ERRORLEVEL=1"
        GOTO EndOfRunAgentNoPop 
    )

    ECHO [!SCRIPT_NAME!] Current directory: "%cd%"
    ECHO [!SCRIPT_NAME!] Executing: python "!PYTHON_MAIN_SCRIPT!" --llm_provider ollama %*
    ECHO.
    python "!PYTHON_MAIN_SCRIPT!" --llm_provider ollama %*
    SET "PYTHON_EXEC_ERRORLEVEL=!ERRORLEVEL!"
    ECHO.
    ECHO [!SCRIPT_NAME!] Python script finished. Errorlevel from Python: !PYTHON_EXEC_ERRORLEVEL!
    POPD
    ECHO [!SCRIPT_NAME!] Restored original directory. Current directory: "%cd%"
    GOTO EndOfRunAgentLogicComplete

:EndOfRunAgentNoPop
    ECHO [!SCRIPT_NAME!] Error occurred before or during PUSHD.
:EndOfRunAgentLogicComplete
    REM Removed IF/ELSE block that was printing success/failure messages here
    ECHO [DEBUG RunAgent] Preparing to exit :RunAgent with code: !PYTHON_EXEC_ERRORLEVEL!
    ENDLOCAL & (
        EXIT /B %PYTHON_EXEC_ERRORLEVEL%
    )

REM REMOVED :HandleExit subroutine as its main purpose was message reporting based on errorlevel