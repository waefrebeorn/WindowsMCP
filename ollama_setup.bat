@echo OFF
REM This script assists in setting up Ollama for use with WuBu.
REM It checks for Ollama installation, provides guidance, and downloads recommended models.
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_EXIT_CODE=0"
SET "SCRIPT_NAME=ollama_setup.bat"

REM --- Main Script ---
ECHO [!SCRIPT_NAME!] Starting Ollama setup for WuBu...
CALL :CheckOllama
IF "!OLLAMA_INSTALLED!"=="false" (
    CALL :InstallGuide
    CALL :CheckOllama
    IF "!OLLAMA_INSTALLED!"=="false" (
        ECHO.
        ECHO [!SCRIPT_NAME!] Ollama still not found after installation attempt.
        ECHO [!SCRIPT_NAME!] Please try restarting your terminal/command prompt, or ensure Ollama is correctly added to your system PATH.
        CHOICE /C YN /M "[!SCRIPT_NAME!] Do you want to try downloading models anyway (Y/N)?"
        IF ERRORLEVEL 2 (
            ECHO [!SCRIPT_NAME!] Exiting setup. Please resolve Ollama installation issues before running WuBu with Ollama.
            SET "SCRIPT_EXIT_CODE=1"
            GOTO :HandleExit
        )
    )
)

CALL :DownloadModels
REM SCRIPT_EXIT_CODE for DownloadModels is implicitly handled as it doesn't force exit on model pull failure.

ECHO.
ECHO [!SCRIPT_NAME!] Ollama model setup process complete.
ECHO [!SCRIPT_NAME!] You can now try running WuBu.
ECHO [!SCRIPT_NAME!] Use 'python main.py --llm_provider ollama' or a relevant '.bat' script like 'run_ollama_wubu.bat'.
ECHO [!SCRIPT_NAME!] Ensure the models specified in 'config.json' (e.g., OLLAMA_DEFAULT_MODEL, OLLAMA_MOONDREAM_MODEL) are among those pulled.
GOTO :HandleExit

REM --- Subroutine: CheckOllama ---
:CheckOllama
    ECHO.
    ECHO [!SCRIPT_NAME!] Checking for Ollama installation...
    ollama --version >NUL 2>NUL
    IF !ERRORLEVEL! EQU 0 (
        ECHO [!SCRIPT_NAME!] Ollama is installed.
        SET "OLLAMA_INSTALLED=true"
    ) ELSE (
        ECHO [!SCRIPT_NAME!] Ollama not found on system PATH.
        SET "OLLAMA_INSTALLED=false"
    )
EXIT /B 0

REM --- Subroutine: InstallGuide ---
:InstallGuide
    ECHO.
    ECHO [!SCRIPT_NAME!] Ollama installation guide:
    ECHO [!SCRIPT_NAME!] 1. Download Ollama for Windows from: https://ollama.com/download
    ECHO [!SCRIPT_NAME!] 2. Run the installer and follow its instructions.
    ECHO [!SCRIPT_NAME!] 3. Ensure Ollama is running after installation (it usually starts automatically and appears in the system tray).
    ECHO.
    PAUSE "[!SCRIPT_NAME!] Press any key to continue after installing Ollama..."
EXIT /B 0

REM --- Subroutine: DownloadModels ---
:DownloadModels
    ECHO.
    ECHO [!SCRIPT_NAME!] Attempting to download recommended Ollama models for WuBu...
    ECHO [!SCRIPT_NAME!] This may take some time depending on your internet connection and model sizes.
    ECHO.

    SET "MODEL_DEFAULT_LLM=qwen2.5-coder:7b-instruct-q4_K_M"
    SET "MODEL_VISION=moondream"

    ECHO [!SCRIPT_NAME!] Pulling default LLM: "!MODEL_DEFAULT_LLM!" (recommended for OLLAMA_DEFAULT_MODEL due to tool calling capability)
    ollama pull "!MODEL_DEFAULT_LLM!"
    IF !ERRORLEVEL! EQU 0 (
        ECHO [!SCRIPT_NAME!] Successfully pulled "!MODEL_DEFAULT_LLM!".
    ) ELSE (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to pull "!MODEL_DEFAULT_LLM!". Please check your internet connection, ensure Ollama is running, and that the model name is correct. (Error Code: !ERRORLEVEL!)
        REM Not setting SCRIPT_EXIT_CODE=1 here to allow script to continue trying other models or finish.
    )
    ECHO.

    ECHO [!SCRIPT_NAME!] Pulling vision model: "!MODEL_VISION!" (used for OLLAMA_MOONDREAM_MODEL)
    ollama pull "!MODEL_VISION!"
    IF !ERRORLEVEL! EQU 0 (
        ECHO [!SCRIPT_NAME!] Successfully pulled "!MODEL_VISION!".
    ) ELSE (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to pull "!MODEL_VISION!". (Error Code: !ERRORLEVEL!)
    )
    ECHO.
EXIT /B 0

REM --- Zonos TTS (eSpeak NG) guidance has been moved to setup_venv.bat ---
REM Kept the label here in case of old references, but it does nothing now.
:ZonosGuidance
    REM ECHO [!SCRIPT_NAME!] Zonos TTS guidance is now part of setup_venv.bat (Docker setup).
EXIT /B 0


REM === Final Exit Point ===
:HandleExit
    REM CALL :ZonosGuidance REM No longer calling this here. setup_venv.bat handles Docker/Zonos prereqs.
    ECHO.
IF "!SCRIPT_EXIT_CODE!" NEQ "0" (
    ECHO [!SCRIPT_NAME!] Script finished with errors. Error Code: !SCRIPT_EXIT_CODE!
) ELSE (
    ECHO [!SCRIPT_NAME!] Script finished.
)
ENDLOCAL & (
    PAUSE
    EXIT /B %SCRIPT_EXIT_CODE%
)
