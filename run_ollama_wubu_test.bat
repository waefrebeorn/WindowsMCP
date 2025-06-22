@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_EXIT_CODE=0"

REM === Configuration ===
SET "VENV_DIR=venv"
SET "AGENT_SCRIPT=main.py"
SET "AGENT_NAME=WuBu"
SET "SCRIPT_NAME=run_ollama_wubu_test.bat"

REM Default model for testing. Can be overridden by editing this script.
SET "OLLAMA_TEST_MODEL=qwen2.5-coder:7b-instruct-q4_K_M" 
REM The test command file used by main.py's --test_file argument
SET "TEST_COMMAND_FILE=test_commands.txt"

ECHO [!SCRIPT_NAME!] Starting !AGENT_NAME! Test Script (Ollama Provider)...
ECHO [!SCRIPT_NAME!] WuBu will index files in the current directory: "%cd%"

REM === Activate Virtual Environment ===
CALL :ActivateVenv
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Failed to activate virtual environment.
    SET "SCRIPT_EXIT_CODE=1"
    GOTO :HandleExit
)

REM === Check Ollama Installation ===
CALL :CheckOllamaInstallation
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Ollama installation check failed.
    SET "SCRIPT_EXIT_CODE=1"
    GOTO :HandleExit
)

REM === Check Ollama Service ===
CALL :CheckOllamaService
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] WARNING: Ollama service check indicated service is not responding.
    ECHO [!SCRIPT_NAME!] Please ensure Ollama is running manually. The script will attempt to run the agent, but it may fail.
    REM No GOTO HandleExit here, allow user to proceed if they wish, Python script will show connection error.
)

REM === Check for Test Command File ===
ECHO [!SCRIPT_NAME!] Looking for test command file: "%~dp0!TEST_COMMAND_FILE!"
IF NOT EXIST "%~dp0!TEST_COMMAND_FILE!" (
    ECHO [!SCRIPT_NAME!] ERROR: Test command file "!TEST_COMMAND_FILE!" not found in "%~dp0".
    ECHO [!SCRIPT_NAME!] Please create this file or check the name.
    SET "SCRIPT_EXIT_CODE=1"
    GOTO :HandleExit
)
ECHO [!SCRIPT_NAME!] Test command file found.

REM === Run the Agent with Test File ===
ECHO [!SCRIPT_NAME!] Starting the !AGENT_NAME! with Ollama model "!OLLAMA_TEST_MODEL!" using test file "%~dp0!TEST_COMMAND_FILE!"...
ECHO [!SCRIPT_NAME!] Command: python "%~dp0!AGENT_SCRIPT!" --llm_provider ollama --ollama_model "!OLLAMA_TEST_MODEL!" --test_file "%~dp0!TEST_COMMAND_FILE!"
python "%~dp0!AGENT_SCRIPT!" --llm_provider ollama --ollama_model "!OLLAMA_TEST_MODEL!" --test_file "%~dp0!TEST_COMMAND_FILE!"
SET "AGENT_ERRORLEVEL=!ERRORLEVEL!"

IF !AGENT_ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] WARNING: !AGENT_NAME! script exited with error code !AGENT_ERRORLEVEL!.
    SET "SCRIPT_EXIT_CODE=!AGENT_ERRORLEVEL!"
) ELSE (
    ECHO [!SCRIPT_NAME!] !AGENT_NAME! script completed successfully.
)

GOTO :HandleExit

REM === Subroutines ===
:ActivateVenv
    ECHO [!SCRIPT_NAME!] Activating Python virtual environment from "%~dp0!VENV_DIR!"...
    IF NOT EXIST "%~dp0!VENV_DIR!\Scripts\activate.bat" (
        ECHO [!SCRIPT_NAME!] ERROR: Virtual environment activation script not found at "%~dp0!VENV_DIR!\Scripts\activate.bat".
        ECHO [!SCRIPT_NAME!] Please run 'setup_venv.bat' first.
        EXIT /B 1
    )
    CALL "%~dp0!VENV_DIR!\Scripts\activate.bat"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to execute activate.bat for the virtual environment.
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Python virtual environment activated.
EXIT /B 0

:CheckOllamaInstallation
    ECHO [!SCRIPT_NAME!] Checking for Ollama installation...
    ollama --version >nul 2>&1
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Ollama is not installed or not found in PATH.
        ECHO [!SCRIPT_NAME!] Please install Ollama (e.g., run ollama_setup.bat or from https://ollama.com) and ensure it's in your PATH.
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Ollama is installed.
EXIT /B 0

:CheckOllamaService
    ECHO [!SCRIPT_NAME!] Checking if Ollama service is responding...
    ollama list >nul 2>nul
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] Ollama service does not appear to be responding.
        ECHO [!SCRIPT_NAME!] The Python script will attempt to connect but may fail if Ollama is not started.
        REM Return errorlevel 1 to indicate service not ready, but main script can decide to proceed.
        EXIT /B 1
    ) ELSE (
        ECHO [!SCRIPT_NAME!] Ollama service is responding.
    )
EXIT /B 0

:HandleExit
    ECHO [!SCRIPT_NAME!] Script finished. Exit Code: !SCRIPT_EXIT_CODE!
    ENDLOCAL & (
        PAUSE
        EXIT /B %SCRIPT_EXIT_CODE%
    )
