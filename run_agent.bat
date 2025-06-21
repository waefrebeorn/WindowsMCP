@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_EXIT_CODE=0"

REM === Configuration ===
SET "VENV_DIR=venv"
SET "AGENT_SCRIPT=main.py"
SET "AGENT_NAME=Windows Desktop AI Assistant"

REM === Main Logic ===
ECHO Starting %AGENT_NAME%...
CALL :ActivateVenv
IF !ERRORLEVEL! NEQ 0 (
    ECHO ERROR: Failed to activate virtual environment.
    SET "SCRIPT_EXIT_CODE=1"
    GOTO :HandleExit
)

CALL :RunAgent
REM The SCRIPT_EXIT_CODE from RunAgent will propagate
SET "SCRIPT_EXIT_CODE=!ERRORLEVEL!"
IF !SCRIPT_EXIT_CODE! NEQ 0 (
    ECHO INFO: %AGENT_NAME% script exited with error code !SCRIPT_EXIT_CODE!.
) ELSE (
    ECHO %AGENT_NAME% script completed.
)

GOTO :HandleExit

REM === Subroutines ===
:ActivateVenv
    ECHO.
    ECHO Activating Python virtual environment from '!VENV_DIR!'...
    IF NOT EXIST "%~dp0!VENV_DIR!\Scripts\activate.bat" (
        ECHO ERROR: Virtual environment activation script not found at '%~dp0!VENV_DIR!\Scripts\activate.bat'.
        ECHO Please run the 'setup_venv.bat' script first to create the virtual environment and install dependencies.
        EXIT /B 1
    )
    CALL "%~dp0!VENV_DIR!\Scripts\activate.bat"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO ERROR: Failed to execute activate.bat script for the virtual environment.
        EXIT /B 1
    )
    ECHO Virtual environment activated.
EXIT /B 0

:RunAgent
    ECHO.
    ECHO Starting the %AGENT_NAME% in Interactive Mode ('%~dp0!AGENT_SCRIPT!')...
    ECHO You can pass arguments like --voice, --llm_provider, etc. by editing this script or running main.py directly.
    IF NOT EXIST "%~dp0!AGENT_SCRIPT!" (
        ECHO ERROR: Agent script '%~dp0!AGENT_SCRIPT!' not found.
        EXIT /B 1
    )
    REM Add any default arguments for the agent here. For example:
    REM python "%~dp0!AGENT_SCRIPT!" --llm_provider ollama
    python "%~dp0!AGENT_SCRIPT!"
    SET "PYTHON_ERRORLEVEL=!ERRORLEVEL!"
    REM No pause here, main.py will keep running in interactive mode.

    IF !PYTHON_ERRORLEVEL! NEQ 0 (
        ECHO WARNING: The %AGENT_NAME% script exited with error code !PYTHON_ERRORLEVEL!.
    )
EXIT /B !PYTHON_ERRORLEVEL!

REM === Final Exit Point ===
:HandleExit
ENDLOCAL & (
    ECHO.
    IF %SCRIPT_EXIT_CODE% NEQ 0 (
        ECHO %AGENT_NAME% execution finished with errors (Code: %SCRIPT_EXIT_CODE%).
    ) ELSE (
        ECHO %AGENT_NAME% execution finished.
    )
    PAUSE
    EXIT /B %SCRIPT_EXIT_CODE%
)
