@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_OVERALL_EXIT_CODE=0"

REM === Configuration ===
SET "SCRIPT_NAME=run_ollama_wubu.bat"
SET "AGENT_NAME=WuBu"
SET "VENV_DIR_NAME=venv"
SET "PYTHON_MAIN_SCRIPT=main.py"
SET "PROJECT_ROOT=%~dp0"
SET "VENV_ACTIVATION_SCRIPT=!PROJECT_ROOT!!VENV_DIR_NAME!\Scripts\activate.bat"
SET "AGENT_SCRIPT_PATH=!PROJECT_ROOT!!PYTHON_MAIN_SCRIPT!"

ECHO [!SCRIPT_NAME!] Starting !AGENT_NAME! with Ollama provider...

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

    CALL :CheckOllamaInstallation
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Ollama installation check failed in :MainLogic.
        SET "SUB_EXIT_CODE=1"
        GOTO :EndOfMainLogicSub
    )

    CALL :CheckOllamaService
    IF !ERRORLEVEL! NEQ 0 (
        ECHO.
        ECHO [!SCRIPT_NAME!] WARNING: The Ollama service does not seem to be responding (Error Code: !ERRORLEVEL!).
        ECHO [!SCRIPT_NAME!] Please ensure the Ollama application/service is running manually.
        CHOICE /C YN /M "[!SCRIPT_NAME!] Do you want to try running the agent anyway (Y/N)?"
        IF ERRORLEVEL 2 (
            ECHO [!SCRIPT_NAME!] Exiting. Agent will not run without Ollama service.
            SET "SUB_EXIT_CODE=1"
            GOTO :EndOfMainLogicSub
        )
        ECHO [!SCRIPT_NAME!] Proceeding despite potential Ollama service issue. Agent may fail to connect to Ollama.
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
:CheckOllamaInstallation
    SETLOCAL
    ECHO.
    ECHO [!SCRIPT_NAME!] Checking for Ollama installation...
    ollama --version >NUL 2>NUL
    IF !ERRORLEVEL! EQU 0 (
        ECHO [!SCRIPT_NAME!] Ollama is installed.
        ENDLOCAL
        EXIT /B 0
    ) ELSE (
        ECHO [!SCRIPT_NAME!] ERROR: Ollama installation not found or not in PATH.
        ECHO [!SCRIPT_NAME!] Please run 'ollama_setup.bat' or install Ollama manually from https://ollama.com.
        ENDLOCAL
        EXIT /B 1
    )

REM ----------------------------------------------------------------------------
:CheckOllamaService
    SETLOCAL
    ECHO.
    ECHO [!SCRIPT_NAME!] Checking if Ollama service is responding...
    ollama list >NUL 2>NUL
    IF !ERRORLEVEL! EQU 0 (
        ECHO [!SCRIPT_NAME!] Ollama service is responding.
        ENDLOCAL
        EXIT /B 0
    ) ELSE (
        ECHO [!SCRIPT_NAME!] Ollama service does not appear to be running or responding.
        ECHO [!SCRIPT_NAME!] The Python script will attempt to connect, but may fail if Ollama is not started.
        ENDLOCAL
        EXIT /B 1 REM Indicate service not ready, main logic will handle user choice.
    )

REM ----------------------------------------------------------------------------
:RunAgent
    SETLOCAL ENABLEDELAYEDEXPANSION
    ECHO.
    ECHO [!SCRIPT_NAME!] Preparing to start the !AGENT_NAME! with Ollama provider.
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
    ECHO [!SCRIPT_NAME!] The agent will use the Ollama model specified in config.json (OLLAMA_DEFAULT_MODEL).
    ECHO [!SCRIPT_NAME!] To use a specific model or other options, provide them as arguments to this script, e.g.:
    ECHO [!SCRIPT_NAME!]   !SCRIPT_NAME! --ollama_model "your_model_name_here" --voice
    ECHO [!SCRIPT_NAME!] Executing: python "!PYTHON_MAIN_SCRIPT!" --llm_provider ollama %*
    ECHO.

    python "!PYTHON_MAIN_SCRIPT!" --llm_provider ollama %*
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
        ECHO [!SCRIPT_NAME!] !AGENT_NAME! (Ollama) execution finished successfully.
    ) ELSE (
        ECHO [!SCRIPT_NAME!] !AGENT_NAME! (Ollama) execution finished with ERRORLEVEL: !EXIT_CODE_TO_USE!.
        ECHO [!SCRIPT_NAME!] Please review the output above for details.
    )

    REM PAUSE is often used for debugging in a console that closes on exit
    REM PAUSE
    ENDLOCAL
EXIT /B %EXIT_CODE_TO_USE%
