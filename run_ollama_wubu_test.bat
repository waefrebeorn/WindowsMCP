@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "TOPLEVEL_SCRIPT_EXIT_CODE=0"

REM === Configuration ===
SET "SCRIPT_NAME=run_ollama_wubu_test.bat"
SET "AGENT_NAME=WuBu"
SET "VENV_DIR_NAME=venv"
SET "PYTHON_MAIN_SCRIPT=main.py"
SET "PROJECT_ROOT=%~dp0"
SET "VENV_ACTIVATION_SCRIPT=%PROJECT_ROOT%%VENV_DIR_NAME%\Scripts\activate.bat"
SET "AGENT_SCRIPT_PATH=%PROJECT_ROOT%%PYTHON_MAIN_SCRIPT%"

REM Default model for testing.
SET "OLLAMA_DEFAULT_TEST_MODEL=qwen2.5-coder:7b-instruct-q4_K_M"
REM The test command file used by main.py's --test_file argument
SET "DEFAULT_TEST_COMMAND_FILE=test_commands.txt"
SET "TEST_COMMAND_FILE_PATH=%PROJECT_ROOT%%DEFAULT_TEST_COMMAND_FILE%"

ECHO [!SCRIPT_NAME!] Starting !AGENT_NAME! Test Script (Ollama Provider)...

REM Initialize arguments to their defaults
SET "OLLAMA_MODEL_TO_USE=!OLLAMA_DEFAULT_TEST_MODEL!"
SET "TEST_FILE_TO_USE=!TEST_COMMAND_FILE_PATH!"
SET "FORWARDED_ARGS="

REM === Argument Parsing ===
:ArgParseLoop
IF "%~1"=="" GOTO :ArgParseDone
IF /I "%~1"=="--ollama_model" (
    IF "%~2"=="" (
        ECHO [!SCRIPT_NAME!] ERROR: --ollama_model requires a value.
        SET "TOPLEVEL_SCRIPT_EXIT_CODE=1"
        GOTO :FinalExitPointOnlyMessage
    )
    SET "OLLAMA_MODEL_TO_USE=%~2"
    SHIFT
    SHIFT
    GOTO :ArgParseLoop
)
IF /I "%~1"=="--test_file" (
    IF "%~2"=="" (
        ECHO [!SCRIPT_NAME!] ERROR: --test_file requires a value.
        SET "TOPLEVEL_SCRIPT_EXIT_CODE=1"
        GOTO :FinalExitPointOnlyMessage
    )
    SET "TEST_FILE_TO_USE=%~2"
    SHIFT
    SHIFT
    GOTO :ArgParseLoop
)
REM If argument is not recognized by batch, add it to FORWARDED_ARGS
IF DEFINED FORWARDED_ARGS (
    SET "FORWARDED_ARGS=!FORWARDED_ARGS! "%~1""
) ELSE (
    SET "FORWARDED_ARGS="%~1""
)
SHIFT
GOTO :ArgParseLoop
:ArgParseDone

ECHO [DEBUG TopLevel] Effective Ollama Model: "!OLLAMA_MODEL_TO_USE!"
ECHO [DEBUG TopLevel] Effective Test File: "!TEST_FILE_TO_USE!"
ECHO [DEBUG TopLevel] Arguments being forwarded to Python script (beyond specific test args): !FORWARDED_ARGS!

REM === Main Logic Call ===
CALL :MainLogic "!OLLAMA_MODEL_TO_USE!" "!TEST_FILE_TO_USE!" !FORWARDED_ARGS!
SET "TOPLEVEL_SCRIPT_EXIT_CODE=!ERRORLEVEL!"

:FinalExitPointOnlyMessage
REM === Final Exit Point ===
ECHO.
ECHO [!SCRIPT_NAME!] Script execution finished.
ECHO [!SCRIPT_NAME!] Final exit code for this script will be: !TOPLEVEL_SCRIPT_EXIT_CODE!
ECHO [!SCRIPT_NAME!] If this is non-zero, an error occurred. Review Python script output and any messages above.
PAUSE "[!SCRIPT_NAME!] Press any key to exit..."
ENDLOCAL
EXIT /B %TOPLEVEL_SCRIPT_EXIT_CODE%

REM ============================================================================
REM === SUBROUTINES ============================================================
REM ============================================================================

:MainLogic
    SETLOCAL ENABLEDELAYEDEXPANSION
    SET "SUB_EXIT_CODE=0"
    SET "OLLAMA_MODEL_PARAM=%~1"
    SET "TEST_FILE_PARAM=%~2"
    SHIFT
    SHIFT
    SET "OTHER_FORWARDED_ARGS=%*" 

    ECHO [DEBUG MainLogic] Received Ollama Model: "!OLLAMA_MODEL_PARAM!"
    ECHO [DEBUG MainLogic] Received Test File: "!TEST_FILE_PARAM!"
    ECHO [DEBUG MainLogic] Received Other Forwarded Args: !OTHER_FORWARDED_ARGS!

    CALL :ActivateVenv
    IF "!ERRORLEVEL!" NEQ "0" (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to activate virtual environment.
        SET "SUB_EXIT_CODE=1"
        GOTO :EndOfMainLogic_DirectExit
    )

    ECHO [!SCRIPT_NAME!] NOTE: Ollama installation and service checks have been removed.
    ECHO [!SCRIPT_NAME!] User must ensure Ollama is installed and running.

    CALL :CheckTestCommandFile "!TEST_FILE_PARAM!"
    IF "!ERRORLEVEL!" NEQ "0" (
        ECHO [!SCRIPT_NAME!] ERROR: Test command file check failed for "!TEST_FILE_PARAM!".
        SET "SUB_EXIT_CODE=1"
        GOTO :EndOfMainLogic_DirectExit
    )

    CALL :RunAgentWithTestFile "!OLLAMA_MODEL_PARAM!" "!TEST_FILE_PARAM!" !OTHER_FORWARDED_ARGS!
    SET "SUB_EXIT_CODE=!ERRORLEVEL!"
    ECHO [DEBUG MainLogic] Subroutine :RunAgentWithTestFile exited with code: !SUB_EXIT_CODE!

:EndOfMainLogic_DirectExit
    ECHO [DEBUG MainLogic] Exiting :MainLogic with code: !SUB_EXIT_CODE!
    ENDLOCAL & (
        EXIT /B %SUB_EXIT_CODE%
    )

REM ----------------------------------------------------------------------------
:ActivateVenv
    ECHO.
    ECHO [!SCRIPT_NAME!] Activating Python virtual environment...
    ECHO [!SCRIPT_NAME!] Venv path: "!VENV_ACTIVATION_SCRIPT!"
    IF NOT EXIST "!VENV_ACTIVATION_SCRIPT!" (EXIT /B 1)
    CALL "!VENV_ACTIVATION_SCRIPT!"
    IF "!ERRORLEVEL!" NEQ "0" (EXIT /B 1)
    ECHO [!SCRIPT_NAME!] Virtual environment seems activated.
EXIT /B 0

REM ----------------------------------------------------------------------------
:CheckTestCommandFile
    SETLOCAL
    SET "FILE_TO_CHECK=%~1"
    ECHO.
    ECHO [!SCRIPT_NAME!] Looking for test command file: "!FILE_TO_CHECK!"
    IF "%FILE_TO_CHECK%"=="" (EXIT /B 2)
    IF NOT EXIST "!FILE_TO_CHECK!" (EXIT /B 1)
    ECHO [!SCRIPT_NAME!] Test command file found: "!FILE_TO_CHECK!"
    ENDLOCAL
EXIT /B 0

REM ----------------------------------------------------------------------------
:RunAgentWithTestFile
    SETLOCAL ENABLEDELAYEDEXPANSION
    SET "PYTHON_EXEC_ERRORLEVEL=0"
    SET "OLLAMA_MODEL_ARG=%~1"
    SET "TEST_FILE_ARG=%~2"
    SHIFT
    SHIFT
    SET "PASSTHROUGH_ARGS=%*"

    ECHO.
    ECHO [!SCRIPT_NAME!] Preparing to start !AGENT_NAME! (Test Mode)...
    ECHO [!SCRIPT_NAME!]   Model: "!OLLAMA_MODEL_ARG!"
    ECHO [!SCRIPT_NAME!]   Test File: "!TEST_FILE_ARG!"
    ECHO [!SCRIPT_NAME!]   Agent Script: "!AGENT_SCRIPT_PATH!"
    ECHO [!SCRIPT_NAME!]   Passthrough CLI Args: !PASSTHROUGH_ARGS!

    IF NOT EXIST "!AGENT_SCRIPT_PATH!" (SET "PYTHON_EXEC_ERRORLEVEL=1" && GOTO EndOfRunAgentTestNoPop)

    ECHO [!SCRIPT_NAME!] Changing to project root: "!PROJECT_ROOT!"
    PUSHD "!PROJECT_ROOT!"
    IF "!ERRORLEVEL!" NEQ "0" (SET "PYTHON_EXEC_ERRORLEVEL=1" && GOTO EndOfRunAgentTestNoPop)

    ECHO [!SCRIPT_NAME!] Current directory: "%cd%"
    ECHO [!SCRIPT_NAME!] Executing: python "!PYTHON_MAIN_SCRIPT!" --llm_provider ollama --ollama_model "!OLLAMA_MODEL_ARG!" --test_file "!TEST_FILE_ARG!" !PASSTHROUGH_ARGS!
    ECHO.

    python "!PYTHON_MAIN_SCRIPT!" --llm_provider ollama --ollama_model "!OLLAMA_MODEL_ARG!" --test_file "!TEST_FILE_ARG!" !PASSTHROUGH_ARGS!
    SET "PYTHON_EXEC_ERRORLEVEL=!ERRORLEVEL!"
    ECHO.
    ECHO [!SCRIPT_NAME!] Python script finished. Errorlevel from Python: !PYTHON_EXEC_ERRORLEVEL!

    POPD
    ECHO [!SCRIPT_NAME!] Restored original directory. Current directory: "%cd%"
    GOTO EndOfRunAgentTestLogicComplete

:EndOfRunAgentTestNoPop
    ECHO [!SCRIPT_NAME!] Error occurred before or during PUSHD.
:EndOfRunAgentTestLogicComplete
    ECHO [DEBUG RunAgentWithTestFile] Preparing to exit with code: !PYTHON_EXEC_ERRORLEVEL!
    ENDLOCAL & (
        EXIT /B %PYTHON_EXEC_ERRORLEVEL%
    )