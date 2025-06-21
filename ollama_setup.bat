@echo OFF
REM This script assists in setting up Ollama for use with the Windows Desktop AI Assistant.
REM It checks for Ollama installation, provides guidance, and downloads recommended models.
REM This script does not activate the Python virtual environment as it primarily
REM interacts with the system-level 'ollama' command.
SETLOCAL ENABLEDELAYEDEXPANSION

REM --- Main Script ---
echo [OLLAMA SETUP] Starting Ollama setup for the Windows Desktop AI Assistant...
CALL :CheckOllama
IF "!OLLAMA_INSTALLED!"=="false" (
    CALL :InstallGuide
    CALL :CheckOllama
    IF "!OLLAMA_INSTALLED!"=="false" (
        echo.
        echo [OLLAMA SETUP] Ollama still not found after installation attempt.
        echo [OLLAMA SETUP] Please try restarting your terminal/command prompt, or ensure Ollama is correctly added to your system PATH.
        CHOICE /C YN /M "[OLLAMA SETUP] Do you want to try downloading models anyway (Y/N)?"
        IF ERRORLEVEL 2 (
            echo [OLLAMA SETUP] Exiting setup. Please resolve Ollama installation issues before running the assistant with Ollama.
            GOTO :EOF
        )
    )
)

CALL :DownloadModels

echo.
echo [OLLAMA SETUP] Ollama model setup process complete.
echo [OLLAMA SETUP] You can now try running the Windows Desktop AI Assistant.
echo [OLLAMA SETUP] Use 'python main.py --llm_provider ollama' or a relevant '.bat' script like 'run_ollama_agent.bat'.
echo [OLLAMA SETUP] Ensure the models specified in 'config.json' (e.g., OLLAMA_DEFAULT_MODEL, OLLAMA_MOONDREAM_MODEL) are among those pulled.
GOTO :EOF

REM --- Subroutine: CheckOllama ---
:CheckOllama
echo.
echo [OLLAMA SETUP] Checking for Ollama installation...
ollama --version >NUL 2>NUL
IF !ERRORLEVEL! EQU 0 (
    echo [OLLAMA SETUP] Ollama is installed.
    SET OLLAMA_INSTALLED=true
) ELSE (
    echo [OLLAMA SETUP] Ollama not found on system PATH.
    SET OLLAMA_INSTALLED=false
)
GOTO :EOF

REM --- Subroutine: InstallGuide ---
:InstallGuide
echo.
echo [OLLAMA SETUP] Ollama installation guide:
echo [OLLAMA SETUP] 1. Download Ollama for Windows from: https://ollama.com/download
echo [OLLAMA SETUP] 2. Run the installer and follow its instructions.
echo [OLLAMA SETUP] 3. Ensure Ollama is running after installation (it usually starts automatically and appears in the system tray).
echo.
pause "[OLLAMA SETUP] Press any key to continue after installing Ollama..."
GOTO :EOF

REM --- Subroutine: DownloadModels ---
:DownloadModels
echo.
echo [OLLAMA SETUP] Attempting to download recommended Ollama models for the Desktop AI Assistant...
echo [OLLAMA SETUP] This may take some time depending on your internet connection and model sizes.
echo.

SET MODEL_DEFAULT_LLM=qwen2.5-coder:7b-instruct-q4_K_M
SET MODEL_VISION=moondream

echo [OLLAMA SETUP] Pulling default LLM: !MODEL_DEFAULT_LLM! (recommended for OLLAMA_DEFAULT_MODEL due to tool calling capability)
ollama pull "!MODEL_DEFAULT_LLM!"
IF !ERRORLEVEL! EQU 0 (
    echo [OLLAMA SETUP] Successfully pulled !MODEL_DEFAULT_LLM!.
) ELSE (
    echo [OLLAMA SETUP] ERROR: Failed to pull !MODEL_DEFAULT_LLM!. Please check your internet connection, ensure Ollama is running, and that the model name is correct. (Error Code: !ERRORLEVEL!)
)
echo.

echo [OLLAMA SETUP] Pulling vision model: !MODEL_VISION! (used for OLLAMA_MOONDREAM_MODEL)
ollama pull "!MODEL_VISION!"
IF !ERRORLEVEL! EQU 0 (
    echo [OLLAMA SETUP] Successfully pulled !MODEL_VISION!.
) ELSE (
    echo [OLLAMA SETUP] ERROR: Failed to pull !MODEL_VISION!. (Error Code: !ERRORLEVEL!)
)
echo.
GOTO :EOF

