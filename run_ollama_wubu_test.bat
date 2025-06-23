@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_OVERALL_EXIT_CODE=0"

REM === Configuration ===
SET "SCRIPT_NAME=run_ollama_wubu_test.bat"
SET "AGENT_NAME=WuBu"
SET "VENV_DIR_NAME=venv"
SET "PYTHON_MAIN_SCRIPT=main.py"
SET "PROJECT_ROOT=%~dp0"
SET "VENV_ACTIVATION_SCRIPT=!PROJECT_ROOT!!VENV_DIR_NAME!\Scripts\activate.bat"
SET "AGENT_SCRIPT_PATH=!PROJECT_ROOT!!PYTHON_MAIN_SCRIPT!"

REM Default model for testing. Can be overridden by editing this script or via command line.
SET "OLLAMA_DEFAULT_TEST_MODEL=qwen2.5-coder:7b-instruct-q4_K_M"
REM The test command file used by main.py's --test_file argument
SET "DEFAULT_TEST_COMMAND_FILE=test_commands.txt"
SET "TEST_COMMAND_FILE_PATH=!PROJECT_ROOT!!DEFAULT_TEST_COMMAND_FILE!"

ECHO [!SCRIPT_NAME!] Starting !AGENT_NAME! Test Script (Ollama Provider)...

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
    SET "OLLAMA_MODEL_TO_USE=!OLLAMA_DEFAULT_TEST_MODEL!"
    SET "TEST_FILE_TO_USE=!TEST_COMMAND_FILE_PATH!"

    REM Basic argument parsing to allow overriding test model or test file
    REM This is a simple parser; for complex args, Python's argparse is better.
    IF NOT "%~1"=="" (
        IF /I "%~1"=="--ollama_model" (
            SET "OLLAMA_MODEL_TO_USE=%~2"
            SHIFT
            SHIFT
        ) ELSE IF /I "%~1"=="--test_file" (
            SET "TEST_FILE_TO_USE=%~2"
            SHIFT
            SHIFT
        )
    )


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
        ECHO [!SCRIPT_NAME!] WARNING: Ollama service check indicated service is not responding.
        ECHO [!SCRIPT_NAME!] Please ensure Ollama is running manually. The script will attempt to run the agent, but it may fail.
        REM No GOTO EndOfMainLogicSub here, allow user to proceed if they wish, Python script will show connection error.
    )

    CALL :CheckTestCommandFile "!TEST_FILE_TO_USE!"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Test command file check failed.
        SET "SUB_EXIT_CODE=1"
        GOTO :EndOfMainLogicSub
    )

    CALL :RunAgentWithTestFile "!OLLAMA_MODEL_TO_USE!" "!TEST_FILE_TO_USE!" %*
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
        ECHO [!SCRIPT_NAME!] Please install Ollama (e.g., run ollama_setup.bat or from https://ollama.com) and ensure it's in your PATH.
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
        ECHO [!SCRIPT_NAME!] Ollama service does not appear to be responding.
        ECHO [!SCRIPT_NAME!] The Python script will attempt to connect but may fail if Ollama is not started.
        ENDLOCAL
        REM Return errorlevel 1 to indicate service not ready, but main script can decide to proceed.
        EXIT /B 1
    )

REM ----------------------------------------------------------------------------
:CheckTestCommandFile
    SETLOCAL
    SET "FILE_TO_CHECK=%~1"
    ECHO.
    ECHO [!SCRIPT_NAME!] Looking for test command file: "!FILE_TO_CHECK!"
    IF NOT EXIST "!FILE_TO_CHECK!" (
        ECHO [!SCRIPT_NAME!] ERROR: Test command file "!FILE_TO_CHECK!" not found.
        ECHO [!SCRIPT_NAME!] Please create this file or check the path/name.
        ENDLOCAL
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Test command file found: "!FILE_TO_CHECK!"
    ENDLOCAL
EXIT /B 0

REM ----------------------------------------------------------------------------
:RunAgentWithTestFile
    SETLOCAL ENABLEDELAYEDEXPANSION
    SET "OLLAMA_MODEL_PARAM=%~1"
    SET "TEST_FILE_PARAM=%~2"
    SHIFT
    SHIFT
    SET "OTHER_ARGS=%*"

    ECHO.
    ECHO [!SCRIPT_NAME!] Preparing to start the !AGENT_NAME! with Ollama model "!OLLAMA_MODEL_PARAM!" using test file "!TEST_FILE_PARAM!"
    ECHO [!SCRIPT_NAME!] Agent script: "!AGENT_SCRIPT_PATH!"
    ECHO [!SCRIPT_NAME!] Other arguments: !OTHER_ARGS!

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
    ECHO [!SCRIPT_NAME!] Executing: python "!PYTHON_MAIN_SCRIPT!" --llm_provider ollama --ollama_model "!OLLAMA_MODEL_PARAM!" --test_file "!TEST_FILE_PARAM!" !OTHER_ARGS!
    ECHO.

    python "!PYTHON_MAIN_SCRIPT!" --llm_provider ollama --ollama_model "!OLLAMA_MODEL_PARAM!" --test_file "!TEST_FILE_PARAM!" !OTHER_ARGS!
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
        ECHO [!SCRIPT_NAME!] !AGENT_NAME! (Ollama Test) execution finished successfully.
    ) ELSE (
        ECHO [!SCRIPT_NAME!] !AGENT_NAME! (Ollama Test) execution finished with ERRORLEVEL: !EXIT_CODE_TO_USE!.
        ECHO [!SCRIPT_NAME!] Please review the output above for details.
    )

    REM PAUSE is often used for debugging in a console that closes on exit
    REM PAUSE
    ENDLOCAL
EXIT /B %EXIT_CODE_TO_USE%
