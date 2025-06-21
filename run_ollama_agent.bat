@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_EXIT_CODE=0"

REM --- Configuration ---
SET "VENV_DIR=venv"
SET "AGENT_SCRIPT=main.py"
SET "AGENT_NAME=Windows Desktop AI Assistant"
SET "SCRIPT_NAME=run_ollama_agent.bat"

REM === Main Logic ===
ECHO Starting %AGENT_NAME% with Ollama provider...

CALL :ActivateVenv
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Failed to activate virtual environment.
    SET "SCRIPT_EXIT_CODE=1"
    GOTO :HandleExit
)

CALL :CheckOllamaInstallation
IF "!OLLAMA_READY!"=="false" (
    SET "SCRIPT_EXIT_CODE=1"
    GOTO :HandleExit
)

CALL :CheckOllamaService
IF "!OLLAMA_SERVICE_READY!"=="false" (
    echo.
    echo [!SCRIPT_NAME!] WARNING: The Ollama service does not seem to be responding.
    echo [!SCRIPT_NAME!] Please ensure the Ollama application/service is running manually.
    CHOICE /C YN /M "[!SCRIPT_NAME!] Do you want to try running the agent anyway (Y/N)?"
    IF ERRORLEVEL 2 (
        echo [!SCRIPT_NAME!] Exiting. Agent will not run without Ollama service.
        SET "SCRIPT_EXIT_CODE=1"
        GOTO :HandleExit
    )
    echo [!SCRIPT_NAME!] Proceeding despite potential Ollama service issue. Agent may fail to connect to Ollama.
)

CALL :RunAgent
SET "SCRIPT_EXIT_CODE=!ERRORLEVEL!"

GOTO :HandleExit

REM --- Subroutine: ActivateVenv ---
:ActivateVenv
    ECHO.
    ECHO [!SCRIPT_NAME!] Activating Python virtual environment from '!VENV_DIR!'...
    IF NOT EXIST "%~dp0!VENV_DIR!\Scripts\activate.bat" (
        ECHO [!SCRIPT_NAME!] ERROR: Virtual environment activation script not found at '%~dp0!VENV_DIR!\Scripts\activate.bat'.
        ECHO [!SCRIPT_NAME!] Please run 'setup_venv.bat' first to create the virtual environment and install dependencies.
        EXIT /B 1
    )
    CALL "%~dp0!VENV_DIR!\Scripts\activate.bat"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to execute activate.bat script for the virtual environment.
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Python virtual environment activated.
EXIT /B 0

REM --- Subroutine: CheckOllamaInstallation ---
:CheckOllamaInstallation
    SET OLLAMA_READY=false
    echo.
    echo [!SCRIPT_NAME!] Checking for Ollama installation...
    ollama --version >NUL 2>NUL
    IF !ERRORLEVEL! EQU 0 (
        echo [!SCRIPT_NAME!] Ollama is installed.
        SET OLLAMA_READY=true
    ) ELSE (
        echo [!SCRIPT_NAME!] ERROR: Ollama installation not found or not in PATH.
        echo [!SCRIPT_NAME!] Please run 'ollama_setup.bat' or install Ollama manually from https://ollama.com.
    )
EXIT /B !ERRORLEVEL!

REM --- Subroutine: CheckOllamaService ---
:CheckOllamaService
    SET OLLAMA_SERVICE_READY=false
    echo.
    echo [!SCRIPT_NAME!] Checking if Ollama service is responding...
    ollama list >NUL 2>NUL
    IF !ERRORLEVEL! EQU 0 (
        echo [!SCRIPT_NAME!] Ollama service is responding.
        SET OLLAMA_SERVICE_READY=true
    ) ELSE (
        echo [!SCRIPT_NAME!] Ollama service does not appear to be running or responding.
        echo [!SCRIPT_NAME!] The Python script will attempt to connect, but may fail if Ollama is not started.
        REM No attempt to start it here, user should manage the Ollama service/application.
    )
EXIT /B 0


REM --- Subroutine: RunAgent ---
:RunAgent
    ECHO.
    ECHO [!SCRIPT_NAME!] Starting the %AGENT_NAME% with Ollama provider.
    ECHO [!SCRIPT_NAME!] The agent will use the Ollama model specified in config.json (OLLAMA_DEFAULT_MODEL).
    ECHO [!SCRIPT_NAME!] To use a specific model with this script, edit the command below, e.g.:
    ECHO [!SCRIPT_NAME!] python "%~dp0%AGENT_SCRIPT%" --llm_provider ollama --ollama_model "your_model_name_here"
    ECHO.
    python "%~dp0%AGENT_SCRIPT%" --llm_provider ollama
    SET "PYTHON_ERRORLEVEL=!ERRORLEVEL!"

    IF !PYTHON_ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] WARNING: The %AGENT_NAME% script exited with error code !PYTHON_ERRORLEVEL!.
    ) ELSE (
        ECHO [!SCRIPT_NAME!] %AGENT_NAME% script completed.
    )
EXIT /B !PYTHON_ERRORLEVEL!

REM === Final Exit Point ===
:HandleExit
ENDLOCAL & (
    ECHO.
    IF %SCRIPT_EXIT_CODE% NEQ 0 (
        ECHO [!SCRIPT_NAME!] %AGENT_NAME% (Ollama) execution finished with errors (Code: %SCRIPT_EXIT_CODE%).
    ) ELSE (
        ECHO [!SCRIPT_NAME!] %AGENT_NAME% (Ollama) execution finished.
    )
    PAUSE
    EXIT /B %SCRIPT_EXIT_CODE%
)

