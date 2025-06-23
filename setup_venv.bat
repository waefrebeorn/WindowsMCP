@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_EXIT_CODE=0"

REM === Configuration ===
SET "VENV_DIR=venv"
SET "REQUIREMENTS_FILE=requirements.txt"
SET "SCRIPT_NAME=setup_venv.bat"

ECHO [!SCRIPT_NAME!] Starting Python Virtual Environment Setup...

REM === Validation ===
CALL :CheckPython
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Python check failed. Script will terminate.
    SET "SCRIPT_EXIT_CODE=!ERRORLEVEL!"
    GOTO :HandleExit
)

CALL :CheckDocker
REM If Docker is not found/working, script continues but Zonos TTS (Docker-based) will not work.
REM User is guided by CheckDocker.

REM === Main Logic ===
CALL :CreateVenv
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Virtual environment creation failed. Script will terminate.
    SET "SCRIPT_EXIT_CODE=!ERRORLEVEL!"
    GOTO :HandleExit
)

CALL :InstallDependencies
IF !ERRORLEVEL! NEQ 0 (
    ECHO [!SCRIPT_NAME!] ERROR: Failed to install dependencies. Script will terminate.
    SET "SCRIPT_EXIT_CODE=!ERRORLEVEL!"
    GOTO :HandleExit
)

REM === Completion (if all successful) ===
ECHO.
ECHO =====================================================================
ECHO [!SCRIPT_NAME!] Setup complete!
ECHO Virtual environment "!VENV_DIR!" is ready and dependencies are installed.
ECHO To activate the virtual environment in your current shell, run:
ECHO   %~dp0!VENV_DIR!\Scripts\activate.bat
ECHO =====================================================================
SET "SCRIPT_EXIT_CODE=0" REM Explicitly set for success path
GOTO :HandleExit


REM === Subroutines ===
:CheckPython
    ECHO [!SCRIPT_NAME!] Checking for Python installation...
    python --version >nul 2>&1
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Python is not installed or not found in your system's PATH.
        ECHO [!SCRIPT_NAME!] Please install Python from python.org and ensure it's added to PATH.
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Python found.
EXIT /B 0

:CheckDocker
    ECHO.
    ECHO [!SCRIPT_NAME!] Checking for Docker (required for Zonos TTS)...
    SET "DOCKER_SETUP_SUCCESS=false" REM Flag to track overall Docker setup for Zonos

    docker --version >NUL 2>NUL
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] WARNING: Docker command not found on system PATH.
        CALL :DockerInstallGuide
        ECHO [!SCRIPT_NAME!] Please re-run this script after installing Docker and ensuring it's in PATH.
        EXIT /B 0 REM Exit this subroutine, main script continues but DOCKER_SETUP_SUCCESS is false
    )

    ECHO [!SCRIPT_NAME!] Docker command is accessible. Checking if Docker engine is responsive...
    docker version >NUL 2>NUL
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Docker command found, but Docker engine is not responsive or `docker version` failed. (Error Code: !ERRORLEVEL!)
        ECHO [!SCRIPT_NAME!] Please ensure Docker Desktop is running correctly and has finished initializing.
        CALL :DockerInstallGuide
        ECHO [!SCRIPT_NAME!] Please re-run this script after ensuring Docker Desktop is fully operational.
        EXIT /B 0 REM Exit this subroutine
    )

    ECHO [!SCRIPT_NAME!] Docker engine is responsive.
    CALL :BuildZonosImageFromSource
    IF !ERRORLEVEL! EQU 0 (
        SET "DOCKER_SETUP_SUCCESS=true"
    ) ELSE (
        ECHO [!SCRIPT_NAME!] ERROR: The Zonos Docker image preparation process failed. Zonos TTS may not function. (Reported Error Code from build: !ERRORLEVEL!)
        REM No need to call DockerInstallGuide here again if Docker itself was responsive. The error is specific to the build.
    )
    REM This subroutine's ERRORLEVEL is implicitly that of the last command (CALL or SET)
    REM If BuildZonosImageFromSource fails, its non-zero ERRORLEVEL will propagate from the CALL.
EXIT /B 0

:DockerInstallGuide
    ECHO.
    ECHO ================================================================================
    ECHO [!SCRIPT_NAME!] IMPORTANT: Docker Desktop Installation Guide (for Zonos TTS)
    ECHO.
    ECHO [!SCRIPT_NAME!] WuBu's Zonos Text-to-Speech engine requires a Docker container.
    ECHO.
    ECHO [!SCRIPT_NAME!] 1. Download Docker Desktop for Windows from the official website:
    ECHO [!SCRIPT_NAME!]    https://www.docker.com/products/docker-desktop/
    ECHO.
    ECHO [!SCRIPT_NAME!] 2. Run the installer and follow the instructions. This might require a system restart.
    ECHO.
    ECHO [!SCRIPT_NAME!] 3. After installation, ensure Docker Desktop is RUNNING.
    ECHO [!SCRIPT_NAME!]    (Look for the Docker whale icon in your system tray and ensure it's not indicating an error).
    ECHO [!SCRIPT_NAME!]    It might take a few minutes for Docker Desktop to fully initialize after starting.
    ECHO.
    ECHO [!SCRIPT_NAME!] 4. IMPORTANT: You may need to restart your Command Prompt or PowerShell
    ECHO [!SCRIPT_NAME!]    (and this script) for the 'docker' command to be recognized and for it to
    ECHO [!SCRIPT_NAME!]    connect to the Docker engine properly.
    ECHO [!SCRIPT_NAME!]    Test by opening a NEW terminal and typing: docker --version
    ECHO [!SCRIPT_NAME!]    If it still fails, Docker Desktop might not be running correctly.
    ECHO ================================================================================
    ECHO.
    PAUSE "[!SCRIPT_NAME!] Press any key to continue after attempting Docker Desktop installation and ensuring it's running..."
EXIT /B 0

:BuildZonosImageFromSource
    ECHO.
    SET "ZONOS_REPO_URL=https://github.com/Zyphra/Zonos.git"
    SET "ZONOS_SRC_DIR=Zonos_src"
    SET "WUBU_ZONOS_IMAGE_TAG=wubu_zonos_image"

    ECHO [!SCRIPT_NAME!] Preparing to build Zonos Docker image (!WUBU_ZONOS_IMAGE_TAG!) from source.
    ECHO [!SCRIPT_NAME!] This involves cloning the Zonos repository and running 'docker build'.
    ECHO [!SCRIPT_NAME!] This may take a significant amount of time and disk space.

    REM Clone Zonos repository if Zonos_src directory doesn't exist
    IF NOT EXIST "%~dp0!ZONOS_SRC_DIR!" (
        ECHO [!SCRIPT_NAME!] Zonos source directory "!ZONOS_SRC_DIR!" not found. Cloning from !ZONOS_REPO_URL!...
        git clone "!ZONOS_REPO_URL!" "!ZONOS_SRC_DIR!"
        SET "GIT_CLONE_ERRORLEVEL=!ERRORLEVEL!"
        IF !GIT_CLONE_ERRORLEVEL! NEQ 0 (
            ECHO [!SCRIPT_NAME!] ERROR: Failed to clone Zonos repository from !ZONOS_REPO_URL!. (Error Code: !GIT_CLONE_ERRORLEVEL!)
            ECHO [!SCRIPT_NAME!] Please check your internet connection, Git installation, and repository URL. Zonos TTS cannot be set up.
            EXIT /B 1
        )
        ECHO [!SCRIPT_NAME!] Zonos repository cloned successfully into "!ZONOS_SRC_DIR!".
    ) ELSE (
        ECHO [!SCRIPT_NAME!] Zonos source directory "!ZONOS_SRC_DIR!" already exists. Using existing directory.
        ECHO [!SCRIPT_NAME!] To ensure you build with the latest Zonos source, you may want to manually delete "!ZONOS_SRC_DIR!" and re-run this script.
    )

    REM Verify Dockerfile exists in the Zonos source directory before attempting to build
    IF NOT EXIST "%~dp0!ZONOS_SRC_DIR!\Dockerfile" (
        ECHO [!SCRIPT_NAME!] ERROR: 'Dockerfile' is missing in the Zonos source directory "%~dp0!ZONOS_SRC_DIR!".
        ECHO [!SCRIPT_NAME!] Cannot build the Zonos Docker image. The Zonos repository might be corrupted or incomplete.
        ECHO [!SCRIPT_NAME!] Try deleting the "%~dp0!ZONOS_SRC_DIR!" directory and re-running this script.
        EXIT /B 1
    )

    ECHO [!SCRIPT_NAME!] Building Zonos Docker image (!WUBU_ZONOS_IMAGE_TAG!) from source in "!ZONOS_SRC_DIR!"...
    ECHO [!SCRIPT_NAME!] This will use the Dockerfile provided within the Zonos repository.
    PUSHD "%~dp0!ZONOS_SRC_DIR!"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Could not change directory to "%~dp0!ZONOS_SRC_DIR!". (Error Code: !ERRORLEVEL!)
        EXIT /B 1
    )

    docker build -t "!WUBU_ZONOS_IMAGE_TAG!" .
    SET "DOCKER_BUILD_ERRORLEVEL=!ERRORLEVEL!"
    POPD
    IF !DOCKER_BUILD_ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to build Zonos Docker image "!WUBU_ZONOS_IMAGE_TAG!" from source. (Error Code: !DOCKER_BUILD_ERRORLEVEL!)
        ECHO [!SCRIPT_NAME!] Please check Docker Desktop is running correctly, and review any errors from the 'docker build' command output above.
        EXIT /B 1
    )

    ECHO [!SCRIPT_NAME!] Successfully built Zonos Docker image: !WUBU_ZONOS_IMAGE_TAG!.
EXIT /B 0

:CreateVenv
    ECHO.
    ECHO [!SCRIPT_NAME!] Checking for virtual environment directory: "!VENV_DIR!"
    IF EXIST "%~dp0!VENV_DIR!\Scripts\activate.bat" (
        ECHO [!SCRIPT_NAME!] Virtual environment "!VENV_DIR!" already exists and appears valid. Skipping creation.
        EXIT /B 0
    )

    IF EXIST "%~dp0!VENV_DIR!" (
        ECHO [!SCRIPT_NAME!] Directory "%~dp0!VENV_DIR!" exists but doesn't seem to be a valid venv.
        ECHO [!SCRIPT_NAME!] You might need to remove it manually if you want to recreate it.
        ECHO [!SCRIPT_NAME!] For now, attempting to proceed assuming it might be usable or will be fixed by Python.
    )

    ECHO [!SCRIPT_NAME!] Creating virtual environment in "%~dp0!VENV_DIR!"...
    python -m venv "%~dp0!VENV_DIR!"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to create virtual environment in "%~dp0!VENV_DIR!".
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Virtual environment created.
EXIT /B 0

:InstallDependencies
    ECHO.
    ECHO [!SCRIPT_NAME!] Activating virtual environment to install dependencies...
    IF NOT EXIST "%~dp0!VENV_DIR!\Scripts\activate.bat" (
        ECHO [!SCRIPT_NAME!] ERROR: Cannot find activate script at "%~dp0!VENV_DIR!\Scripts\activate.bat".
        ECHO [!SCRIPT_NAME!] Virtual environment setup might have failed.
        EXIT /B 1
    )
    CALL "%~dp0!VENV_DIR!\Scripts\activate.bat"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to call the activate script. Venv might be corrupted or not found.
        EXIT /B 1
    )

    ECHO [!SCRIPT_NAME!] Installing dependencies from "%~dp0!REQUIREMENTS_FILE!"...
    IF NOT EXIST "%~dp0!REQUIREMENTS_FILE!" (
        ECHO [!SCRIPT_NAME!] ERROR: "%~dp0!REQUIREMENTS_FILE!" not found. Cannot install dependencies.
        EXIT /B 1
    )
    pip install -r "%~dp0!REQUIREMENTS_FILE!"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to install dependencies from "%~dp0!REQUIREMENTS_FILE!".
        ECHO [!SCRIPT_NAME!] Check your internet connection and the contents of the file.
        EXIT /B 1
    )
    ECHO [!SCRIPT_NAME!] Dependencies installed successfully.

    REM Deactivate the venv after installing dependencies if this script is not meant to keep it active
    ECHO [!SCRIPT_NAME!] Deactivating virtual environment (installation complete).
    CALL :DeactivateVenvQuietly
EXIT /B 0

:DeactivateVenvQuietly
    REM This is a bit of a trick as 'deactivate' is a function defined by activate.bat
    REM and not a standalone script. Calling it directly might not always work as expected
    REM or might output "The system cannot find the path specified." if not careful.
    REM For robustness, it's often better to just let the SETLOCAL/ENDLOCAL handle environment restoration.
    REM However, if an explicit deactivate is desired:
    IF DEFINED VIRTUAL_ENV (
        REM Check if deactivate function exists (it's a bit hacky to check this way)
        REM A more reliable way is usually not needed as ENDLOCAL handles cleanup.
        REM For now, we'll skip a direct "call deactivate" as it can be problematic.
        ECHO [!SCRIPT_NAME!] Venv was active. Script's ENDLOCAL will handle restoration.
    )
EXIT /B 0

REM === Final Exit Point ===
:HandleExit
ECHO.
IF "!SCRIPT_EXIT_CODE!" NEQ "0" (
    ECHO [!SCRIPT_NAME!] Script finished with errors. Error Code: !SCRIPT_EXIT_CODE!
) ELSE (
    ECHO [!SCRIPT_NAME!] Script finished successfully.
)
ENDLOCAL & (
    PAUSE
    EXIT /B %SCRIPT_EXIT_CODE%
)
