@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_OVERALL_EXIT_CODE=0"

REM === Configuration ===
SET "SCRIPT_NAME=run_wubu.bat"
SET "AGENT_NAME=WuBu"
SET "VENV_DIR_NAME=venv"
SET "PYTHON_MAIN_SCRIPT=main.py"
SET "PROJECT_ROOT=%~dp0"
SET "VENV_ACTIVATION_SCRIPT=!PROJECT_ROOT!!VENV_DIR_NAME!\Scripts\activate.bat"
SET "AGENT_SCRIPT_PATH=!PROJECT_ROOT!!PYTHON_MAIN_SCRIPT!"

ECHO [!SCRIPT_NAME!] Starting !AGENT_NAME!...

REM === Main Logic ===
CALL :MainLogic %*
SET "SCRIPT_OVERALL_EXIT_CODE=!ERRORLEVEL!"

REM === Final Exit Point ===
CALL :HandleExit "!SCRIPT_OVERALL_EXIT_CODE!"
ENDLOCAL
EXIT /B %SCRIPT_OVERALL_EXIT_CODE%

REM ============================================================================
REM === SUBROUTINES ============================================================
REM ============================================================================

:MainLogic
    SETLOCAL ENABLEDELAYEDEXPANSION
    SET "SUB_EXIT_CODE=0"

    CALL :ActivateVenv
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to activate virtual environment in :MainLogic.
        SET "SUB_EXIT_CODE=1"
        GOTO :EndOfMainLogicSub
    )

    CALL :RunAgent %*
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] INFO: !AGENT_NAME! script execution reported an error.
        SET "SUB_EXIT_CODE=!ERRORLEVEL!"
    ) ELSE (
        ECHO [!SCRIPT_NAME!] !AGENT_NAME! script completed successfully.
    )

:EndOfMainLogicSub
    ENDLOCAL & SET "SCRIPT_OVERALL_EXIT_CODE=%SUB_EXIT_CODE%"
EXIT /B %SCRIPT_OVERALL_EXIT_CODE%

REM ----------------------------------------------------------------------------
:ActivateVenv
    SETLOCAL
    ECHO.
    ECHO [!SCRIPT_NAME!] Activating Python virtual environment...
    ECHO [!SCRIPT_NAME!] Venv path: "!VENV_ACTIVATION_SCRIPT!"

    IF NOT EXIST "!VENV_ACTIVATION_SCRIPT!" (
        ECHO [!SCRIPT_NAME!] FATAL ERROR: Virtual environment activation script not found.
        ECHO [!SCRIPT_NAME!] Expected at: "!VENV_ACTIVATION_SCRIPT!"
        ECHO [!SCRIPT_NAME!] Please ensure 'setup_venv.bat' (or equivalent) has been run successfully.
        EXIT /B 1
    )

    CALL "!VENV_ACTIVATION_SCRIPT!"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] FATAL ERROR: Failed to CALL the virtual environment activation script.
        ECHO [!SCRIPT_NAME!] Script: "!VENV_ACTIVATION_SCRIPT!"
        EXIT /B 1
    )

    ECHO [!SCRIPT_NAME!] Virtual environment activated.
    ENDLOCAL
EXIT /B 0

REM ----------------------------------------------------------------------------
:RunAgent
    SETLOCAL ENABLEDELAYEDEXPANSION
    ECHO.
    ECHO [!SCRIPT_NAME!] Preparing to start the !AGENT_NAME! script...
    ECHO [!SCRIPT_NAME!] Agent script: "!AGENT_SCRIPT_PATH!"
    ECHO [!SCRIPT_NAME!] Arguments passed: %*

    IF NOT EXIST "!AGENT_SCRIPT_PATH!" (
        ECHO [!SCRIPT_NAME!] FATAL ERROR: The !AGENT_NAME! script ('!PYTHON_MAIN_SCRIPT!') was not found.
        ECHO [!SCRIPT_NAME!] Expected at: "!AGENT_SCRIPT_PATH!"
        EXIT /B 1
    )

    ECHO [!SCRIPT_NAME!] Changing current directory to project root: "!PROJECT_ROOT!"
    PUSHD "!PROJECT_ROOT!"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] FATAL ERROR: Failed to change directory to "!PROJECT_ROOT!".
        EXIT /B 1
    )

    ECHO [!SCRIPT_NAME!] Current directory after PUSHD: "%cd%"
    ECHO [!SCRIPT_NAME!] WuBu will index files in this directory.
    ECHO [!SCRIPT_NAME!] For specific behavior, pass arguments like --voice, --llm_provider, etc.,
    ECHO [!SCRIPT_NAME!] by providing them to this script (e.g., !SCRIPT_NAME! --voice).
    ECHO [!SCRIPT_NAME!] Executing: python "!PYTHON_MAIN_SCRIPT!" %*
    ECHO.

    python "!PYTHON_MAIN_SCRIPT!" %*
    SET "PYTHON_EXEC_ERRORLEVEL=!ERRORLEVEL!"

    ECHO.
    ECHO [!SCRIPT_NAME!] Python script execution finished. Errorlevel: !PYTHON_EXEC_ERRORLEVEL!

    POPD
    ECHO [!SCRIPT_NAME!] Restored original directory. Current directory: "%cd%"

    IF !PYTHON_EXEC_ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] WARNING: The !AGENT_NAME! script ('!PYTHON_MAIN_SCRIPT!') exited with error code !PYTHON_EXEC_ERRORLEVEL!.
        EXIT /B !PYTHON_EXEC_ERRORLEVEL!
    )

    ECHO [!SCRIPT_NAME!] !AGENT_NAME! script execution seems successful.
    ENDLOCAL
EXIT /B 0

REM ----------------------------------------------------------------------------
:HandleExit
    SETLOCAL
    SET "EXIT_CODE_TO_USE=%~1"
    SET "EXIT_CODE_TO_USE=!EXIT_CODE_TO_USE:"=!" REM Remove quotes if any

    ECHO.
    ECHO [!SCRIPT_NAME!] Script finalization.
    IF "!EXIT_CODE_TO_USE!" EQU "0" (
        ECHO [!SCRIPT_NAME!] !AGENT_NAME! execution finished successfully.
    ) ELSE (
        ECHO [!SCRIPT_NAME!] !AGENT_NAME! execution finished with ERRORLEVEL: !EXIT_CODE_TO_USE!.
        ECHO [!SCRIPT_NAME!] Please review the output above for details.
    )

    REM PAUSE is often used for debugging in a console that closes on exit
    REM PAUSE
    ENDLOCAL
EXIT /B %EXIT_CODE_TO_USE%
